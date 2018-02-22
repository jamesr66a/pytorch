#include "Python.h"
#include "interpreter.h"

#include "torch/csrc/autograd/edge.h"
#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/functions/special.h"
#include "torch/csrc/autograd/profiler.h"
#include "torch/csrc/autograd/python_engine.h"
#include "torch/csrc/autograd/python_variable.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/fusion_compiler.h"
#include "torch/csrc/jit/generated/aten_dispatch.h"
#include "torch/csrc/jit/graph_executor.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/pybind.h"
#include "torch/csrc/utils/auto_gil.h"

namespace py = pybind11;

namespace torch { namespace jit {


// Before we translate to intepreter instructions, we do
// some preprocessing of the graph to turn it into a form that is closer
// to what the instructions will look like.
// In particular we:
// * (TODO) desugar Loop trip counts into c = 0, c += 1 instructions in the loop
// * flatten stages so that each stage starts with a load from the stack
//   and ends with a store to the stack
// *. computes move_flags (see Outputs), and inserts
//    'Drop' nodes where necessary to release Tensors after a If/Loop body exits
//    or if a value is never used.
// Outputs are:
// * graph - the post processed copy of g
// * move_flags[n] - a list of booleans, one for each input,
//   indicating whether this is the last use of the value. The interpreter
//   should generate a move rather than a copy in this case.
// * stage_input_types: the type annotations on the inputs to each stage
//   these can be removed once the the backward tracer is no longer used

struct PreprocessGraph {
  PreprocessGraph(Graph & g)
  : graph(g.copy()) {
    desugarTripCounts(graph->block());
    flattenStages();
    dropUnused(graph->block());

    // fill in move_flags by scanning blocks;
    findLastUses(graph->block());
    //TODO: desugar Loop trip counts, for now we drop trip counts
  }
  std::shared_ptr<Graph> graph;
  // for each input, should we move rather than copy the inputs
  std::unordered_map<Node*, std::vector<uint8_t>> move_flags;

  // because JIT classic needs this to fix up gradients, remove when possible
  std::vector<std::vector<TypePtr>> stage_input_types;
private:

  // this currently just _removes_ the trip count inputs and checks they are
  // unused. In the future they will be desugared into normal arithmetic to
  // provide a loop counter
  void desugarTripCounts(Block * b) {
    for(auto n : b->nodes()) {
      if(n->kind() == kLoop) {

        // remove the trip count from Loop inputs, we don't support it yet
        n->removeInput(0);
        JIT_ASSERT(n->blocks()[0]->inputs()[0]->uses().size() == 0 &&
          "NYI - use of trip count variable");
        JIT_ASSERT(n->blocks()[0]->inputs()[1]->uses().size() == 0 &&
          "NYI - use of cond variable in loop");

        // TODO: remove cond as input to loop carries, it just complicates this
        // implementation
        n->blocks()[0]->eraseInput(1);
        n->blocks()[0]->eraseInput(0);
      }
      for(auto sb : n->blocks()) {
        desugarTripCounts(sb);
      }
    }
  }
  Node * prependLocation() {
    //handle corner case where there are no nodes,
    if(graph->nodes().begin() == graph->nodes().end()) {
      return graph->return_node();
    }
    return *graph->nodes().begin();
  }
  // removes all inputs and outputs to a graph, replacing them with nodes before of after each insertStage
  void flattenStages() {
    WithInsertPoint guard(*graph, prependLocation());
    size_t input_pos = 0;
    size_t output_pos = 0;
    auto it = graph->nodes().begin();
    for(size_t i = 0; i <= graph->stage(); i++) {
      stage_input_types.emplace_back();
      auto store = graph->insertNode(graph->create(kStore, 0));
      // TODO: unused inputs need drop nodes added for them.
      while(input_pos < graph->inputs().size() && graph->inputs()[input_pos]->stage() == i) {
        auto nv = store->addOutput();
        auto old_node = graph->inputs()[input_pos];
        stage_input_types[i].push_back(old_node->typeOption());
        old_node->replaceAllUsesWith(nv);
        input_pos++;
      }
      while(it != graph->nodes().end() && it->stage() == i)
        ++it;
      graph->setInsertPoint(*it);
      auto load = graph->insertNode(graph->create(kLoad, 0));
      while(output_pos < graph->outputs().size() && graph->outputs()[output_pos]->stage() == i) {
        load->addInput(graph->outputs()[output_pos]);
        output_pos++;
      }
    }
    while (graph->inputs().size() > 0)
      graph->eraseInput(graph->inputs().size() - 1);
    while (graph->outputs().size() > 0)
      graph->eraseOutput(graph->outputs().size() - 1);
  }

  Node * findOrCreateDropInstructionForNode(Node * n) {
    auto it = drop_for_node.find(n);
    if(it == drop_for_node.end()) {
      auto drop_node = graph->create(kDrop, 0);
      drop_node->insertAfter(n);
      it = drop_for_node.emplace(n, drop_node).first;
    }
    return it->second;
  }

  // finds the node in the _same block_ as 'definition' that contains
  // the 'n' Node
  // prereq: one of n's inputs must be 'definition' to ensure it is in scope
  // e.g.
  // n0: a = 4
  // n1: if <cond>:
  // n2:    b = a + a
  // findNodeInTheSameBlockAs(n2, a) == n1
  Node * findNodeInTheSameBlockAs(Node * n, Value * definition) {
    while(definition->node()->owningBlock() != n->owningBlock()) {
      auto block = n->owningBlock();
      // asserts will fail if 'definition' is not in scope for 'n'
      // use lint to debug
      JIT_ASSERT(block);
      n = block->owningNode();
      JIT_ASSERT(n);
    }
    return n;
  }
  void scanUse(Node * n, size_t i) {
    auto & uses_array = move_flags[n];
    auto v = n->inputs()[i];
    auto inserted = seen.insert(v).second;
    if(!inserted) {
      uses_array[i] = false;
      return;
    }

    // the last use of v may be in a nested block of an If or Loop statement
    // find the node 'same_depth_node' at the same depth as the definition of v,
    // and consider that node to be the last use of v.
    // This ensures we do not delete nodes in nested scopes
    // that may be executed multiple times
    // and that nodes used on one side of an if
    // but not the other get deleted regardless of the branch
    // e.g.
    // a = 4
    // while <...>:
    //   y = a + a
    // drop(a)
    // In other words, we find the first program point for v that
    // _reverse_ dominates the definition of v, and add a drop point there.
    Node * same_depth_node = findNodeInTheSameBlockAs(n, v);

    // In the case where v and n are in the same block, just mark
    // its move_flags to be true
    if(same_depth_node == n) {
      uses_array[i] = true;
      return;
    }

    // in the case where the use is nested in a block
    // add a Drop node after that block which will drop 'v'.
    uses_array[i] = false;
    addToDropIfNotExists(findOrCreateDropInstructionForNode(same_depth_node), v);
  }
  void findLastUses(Node * n) {
    for(auto b : n->blocks()) {
      findLastUses(b);
    }
    move_flags[n].resize(n->inputs().size());
    // scan backwards so if a value is used twice in the list then it is a move
    for(size_t i = n->inputs().size(); i > 0; --i) {
      scanUse(n, i-1);
    }
  }
  void addToDropIfNotExists(Node * drop, Value * v) {
    for(auto i : drop->inputs()) {
      // we already accounted for this use
      if(i == v)
        return;
    }
    drop->addInput(v);
    move_flags[drop].push_back(true);
  }
  void findLastUses(Block * b) {
    findLastUses(b->return_node());
    for(auto n : b->nodes().reverse()) {
      findLastUses(n);
    }
  }
  Node* createDropIfUnused(ArrayRef<Value*> values) {
    std::vector<Value*> to_drop;
    for(auto v : values) {
      if(v->uses().size() == 0)
        to_drop.push_back(v);
    }
    if(to_drop.size() == 0)
      return nullptr;
    return graph->create(kDrop, to_drop, 0);
  }

  // insert Drop nodes to kill references for anything unused:
  // this can happen in a few places, e.g. when a node returns
  // many values but only one is used
  // a, b = foo()
  // return a
  void dropUnused(Block *b) {
    if(auto d = createDropIfUnused(b->inputs())) {
      b->prependNode(d);
    }
    for(auto n : b->nodes()) {
      if(auto d = createDropIfUnused(n->outputs())) {
        d->insertAfter(n);
      }
      for(auto b : n->blocks())
        dropUnused(b);
    }
  }

  // A map from an If or Loop node to the optional Drop block that
  // occurs directly after it to release any tensors that go out of scope
  // when the If/Loop exits. These are created and inserted on demand.
  std::unordered_map<Node*, Node*> drop_for_node;

  // have we seen this value, yet, if not, it is the last use of the value
  std::unordered_set<Value*> seen;
};

// previously the interpreter worked with at::Retainable values,
// which are annoying to handle since 99% of values are at::Tensor anyway
// instead we create a fake subclass of TensorImpl that can be subclassed
// to hold arbitrary things
struct ContainerTensor : public at::TensorImpl {
public:
  ContainerTensor()
  : TensorImpl(&(at::globalContext().getType(at::Backend::Undefined,at::ScalarType::Undefined))) {}

  virtual ~ContainerTensor() {}
  virtual const char * toString() const override {
    throw std::runtime_error("toString() on ContainerTensor");
  }
  virtual at::IntList sizes() const override {
    throw std::runtime_error("sizes() on ContainerTensor");
  }
  virtual at::IntList strides() const override {
    throw std::runtime_error("strides() on ContainerTensor");
  }
  virtual int64_t dim() const override {
    throw std::runtime_error("dim() on ContainerTensor");
  }
  virtual at::Scalar localScalar() override {
    throw std::runtime_error("localScalar() on ContainerTensor");
  }
  virtual void * unsafeGetTH(bool retain) override {
    throw std::runtime_error("unsafeGetTH() on ContainerTensor");
  }
  virtual std::unique_ptr<at::Storage> storage() override {
    throw std::runtime_error("storage() on ContainerTensor");
  }
};


// Dummy function is the last function that the autograd engine calls
// when evaluating Eval nodes. Its input tensors are the outputs that the
// Eval node needs to produce.
// We interscept these values using an Autograd callback. So the function itself
// never runs.
struct DummyFunction : autograd::Function {
  virtual autograd::variable_list apply(const autograd::variable_list& inputs) override {
    throw std::logic_error("DummyFunction::apply() called, but it should be blocked by a callback returning false");
  }
};

// An AutogradHandle holds the information needed to run an Autograd backward pass
// after running a forward operator (such as PythonOp, CppOp, or for double-backwards another Eval Op)
// The EvalOperation uses AutogradHandle to perform this operation.
struct AutogradHandle : public ContainerTensor {

  // The inputs of DummyFunction are the gradients of the forward passes
  // inputs, and the _outputs_ of the run of the Autograd engine computing backward.
  // there is one entry in this list for each forward input that requires
  // gradients
  std::shared_ptr<DummyFunction> forward_inputs;

  // there is one entry in this list for each output of the forward pass
  // that represents the location in the backwaard pass where the gradient
  // of this output should be inserted at the beginning of the backward pass
  autograd::edge_list forward_outputs;
};

// HandleBuilder is used to construct the correct Autograd Handle objects
// for use in a future stage.
// It is used even when the future stage does not require a handle since
// it also performs the conversions between Tensor and Variable, which
// behave differently depending on whether a future handle needs to be
// created.
struct HandleBuilder {
  HandleBuilder(bool requires_handle) {
    if(requires_handle) {
      handle = new AutogradHandle();
      handle->forward_inputs = std::make_shared<DummyFunction>();
    }
  }
  autograd::Variable addInput(at::Tensor && input, const VariableFlags & flags_) {
    if(handle && flags_.requires_grad) {
      auto variable = autograd::make_variable(std::move(input), /*requires_grad=*/false);
      autograd::create_gradient_edge(variable, handle->forward_inputs);
      return variable;
    } else {
      return autograd::make_variable(std::move(input), /*requires_grad=*/false);
    }
  }
  at::Tensor addOutput(const autograd::Variable & output) {
    if(handle) {
      handle->forward_outputs.push_back(output.gradient_edge());
    }
    return output.data();
  }
  void writeTo(Stack & outputs) {
    // outputs takes ownership of handle
    if(handle) {
      outputs.push_back(at::Tensor(handle, /*retain=*/false));
      handle = nullptr;
    }
  }
private:
  AutogradHandle* handle = nullptr;
};

bool hasHandleOutput(Node * n) {
  if(n->outputs().size() == 0)
    return false;
  auto & last = n->outputs().back();
  return last->isHandle() && last->uses().size() > 0; // don't bother creating a handle if it is never used
}

Operation createPythonOperation(PythonOp* op) {
  py::object func = py::handle(op->pyobj.get()).attr("apply");
  bool has_handle = hasHandleOutput(op);
  size_t num_inputs = 0;
  for(auto arg_type : op->cconv) {
    if(arg_type == 't')
      num_inputs++;
  }
  return [=](Stack & stack) {
    AutoGIL gil;
    py::tuple py_inputs(op->cconv.size());
    size_t i = 0;
    size_t next_scalar = 0;
    size_t next_tensor = 0;
    HandleBuilder builder(has_handle);
    for(auto arg_type : op->cconv) {
      if(arg_type == 's') {
        py_inputs[i] = py::reinterpret_borrow<py::object>(op->scalar_args[next_scalar++].get());
      } else if(arg_type == 't') {
        py_inputs[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(
          builder.addInput(std::move(fromLast(stack, num_inputs - next_tensor)), op->var_flags.at(next_tensor))));
        next_tensor++;
      }
      i++;
    }
    drop(stack, num_inputs);
    py::object py_outputs(func(*py_inputs));

    auto addOutput = [&](py::handle entry) {
      if(!THPVariable_Check(entry.ptr())) {
        throw std::runtime_error("Function.apply returned a non-Variable output");
      }
      THPVariable *var = (THPVariable*) entry.ptr();
      stack.push_back(builder.addOutput(var->cdata));
    };
    if(!PyTuple_Check(py_outputs.ptr())) {
      addOutput(py_outputs);
    } else {
      for(py::handle entry : py::tuple(py_outputs)) {
        addOutput(entry);
      }
    }
    builder.writeTo(stack);
    return 0;
  };
}

Operation createCppOperation(CppOp* op) {
  std::shared_ptr<autograd::Function> func = op->fn;
  bool has_handle = hasHandleOutput(op);
  auto num_inputs = op->inputs().size();
  return [=](Stack & stack) {
    HandleBuilder builder(has_handle);
    autograd::variable_list v_inputs;
    for(size_t i = 0; i < num_inputs; i++) {
      v_inputs.push_back(builder.addInput(std::move(fromLast(stack, num_inputs - i)), op->var_flags[i]));
    }
    drop(stack, num_inputs);
    autograd::variable_list v_outputs = (*func)(v_inputs);
    for(auto & output : v_outputs) {
      stack.push_back(builder.addOutput(output));
    }
    builder.writeTo(stack);
    return 0;
  };
}

Operation createEvalOperation(CppOp * op) {
  bool has_handle_output = hasHandleOutput(op);
  auto num_inputs = op->inputs().size();
  return [=](Stack & stack) {
    at::Tensor handle_t = std::move(stack.back());
    AutogradHandle * handle_in = dynamic_cast<AutogradHandle*>(handle_t.get());
    JIT_ASSERT(handle_in);
    HandleBuilder builder(has_handle_output);
    auto& engine = autograd::python::PythonEngine::getDefaultEngine();
    autograd::variable_list v_inputs;
    for(size_t i = 0; i < num_inputs - 1; i++) {
      v_inputs.push_back(builder.addInput(std::move(fromLast(stack, num_inputs - i)), op->var_flags[i]));
    }
    drop(stack, num_inputs);
    // TODO: handle create_graph appropriately
    bool create_graph = true;
    // note: node handle_in->use_count() == 1 means that we are guarenteed that we have the only
    // only copy of the handle. This might make it seem it is ok to pass keep_graph=False.
    // However, it is possible for 'copied_next_fns' to grab functions used by _other_ handles,
    // and these functions will be executed in this run. Since these other handles
    // may still be alive, it is not safe to release the graph
    // TODO: we could cache this list in AutogradHandle (it's read only)
    autograd::edge_list output_edges;
    const auto num_inputs = handle_in->forward_inputs->num_inputs();
    output_edges.reserve(num_inputs);
    for (uint32_t i = 0; i < num_inputs; ++i)
      output_edges.emplace_back(handle_in->forward_inputs, i);
    auto values = engine.execute(handle_in->forward_outputs, v_inputs, true, create_graph, output_edges);
    for(auto & v : values)
      stack.push_back(builder.addOutput(v));
    builder.writeTo(stack);
    return 0;
  };
}

template<typename T>
T tensor_as(at::Tensor&& t) = delete;

template<>
int64_t tensor_as(at::Tensor&& t) {
  // workaround for 1-dim 1-element pytorch tensors until zero-dim
  // tensors are fully supported
  if(t.ndimension() == 1 && t.size(0) == 1) {
    t = t[0];
  }
  return at::Scalar(t).to<int64_t>();
}

// Returns a function implementing functionality of a given node,
// or nullptr if it's a no-op for autograd.
Operation getOperation(jit::Node *node, bool computes_on_variables) {
  IR_IFM(node, PythonOp)
    return createPythonOperation(value);
  IR_ELSEIFM(CppOp)
    if(dynamic_cast<autograd::Eval*>(value->fn.get())) {
      return createEvalOperation(value);
    } else {
      return createCppOperation(value);
    }
  IR_ELSEIF(FusionGroup)
    auto fusion_fn = sharedFusionCompiler().getOrCompile(value);
    auto num_inputs = value->inputs().size();
    return [fusion_fn, num_inputs](Stack & stack) {
      autograd::profiler::RecordFunction record("FusionGroup");
      Stack toutputs;
      // TODO: have fusion_fn work off of a stack as well
      fusion_fn->launch(last(stack, num_inputs), toutputs);
      drop(stack, num_inputs);
      stack.insert(stack.end(), toutputs.begin(), toutputs.end());
      return 0;
    };
  IR_ELSEIF(Constant)
    if(computes_on_variables) {
      auto t = torch::autograd::make_variable(value->t(kvalue), false);
      return [t](Stack & stack) {
        stack.push_back(t);
        return 0;
      };
    } else {
      auto t = value->t(kvalue);
      return [t](Stack & stack) {
        stack.push_back(t);
        return 0;
      };
    }
  IR_ELSEIF(Undefined)
    return [](Stack & stack) {
      stack.push_back(at::Tensor());
      return 0;
    };
  IR_ELSEIF(ReplaceIfUndef)
    return [](Stack & stack) {
      auto alternate = pop(stack);
      auto result = pop(stack);
      if(result.defined()) {
        stack.push_back(std::move(result));
      } else {
        stack.push_back(std::move(alternate));
      }
      return 0;
    };
  IR_ELSEIF(GraphExecutor)
    GraphExecutor executor(value->g(kSubgraph));
    auto num_inputs = value->inputs().size();
    return [=](Stack& stack) mutable {
      autograd::profiler::RecordFunction record("GraphExecutor");
      auto inputs = last(stack, num_inputs);
      variable_tensor_list tinputs(Stack(inputs.begin(), inputs.end()));
      drop(stack, num_inputs);
      //TODO: has graph executor work from a stack as well
      variable_tensor_list toutputs = executor.run(variable_tensor_list(std::move(tinputs)));
      stack.insert(stack.end(), toutputs.begin(), toutputs.end());
      return 0;
    };


  // Load x, y
  // loads values from registers onto the stack, the actual callback does
  // nothing since the stack manipulation is already encoded in inst.inputs
  // and inst.outputs
  IR_ELSEIF(Load)
    return [=](Stack& stack) {
      return 0;
    };

  // x, y = Store
  // stores values from stack into registers, the actual callback does
  // nothing since the stack manipulation is already encoded in inst.inputs
  // and inst.outputs
  IR_ELSEIF(Store)
    return [=](Stack& stack) {
      return 0;
    };
  IR_ELSEIF(Drop)
    auto N = value->inputs().size();
    return [=](Stack& stack) {
      drop(stack, N);
      return 0;
    };
  IR_ELSE()
    return getTensorOp(node).op;
  IR_END()
}


// We need some lists for inputs and outputs. To keep all the memory
// contiguous we allocate a single vector and use offsets into the vector
// which are stored in the ListHandle struct
// start is an offset into int_data of Code for ListHandle<int>
// and bool_data of Code for ListHandle<bool>
template<typename T>
struct ListHandle {
  int start;
  int size;
};

struct UseList {
  // values to be used
  ListHandle<int> values;
  // boolean flags indicating whether to free the Tensor after this use
  ListHandle<bool> free_flags;
};

// one instruction plus meta-data
struct Instruction {
  Operation callback;
  UseList inputs;
  ListHandle<int> outputs;
  Symbol debug_name; // used in dump to understand the generated code
};


int relativeJump(int from_inst, int to_inst) {
  return to_inst - (from_inst + 1);
}

// pre-processing that happens once per graph
struct CodeImpl {
  CodeImpl(std::shared_ptr<Graph> & graph_, bool computes_on_variables)
  : computes_on_variables(computes_on_variables)
  , preprocess(*graph_) {
    graph = preprocess.graph;
    //std::cout << "into code graph:\n" << *graph << "\n";
    insertNodesFromBlock(graph->block());
  }

  // jump when input is 0
  void createJumpZ(int from_inst, int to_inst) {
    auto & inst = instructions[from_inst];
    JIT_ASSERT(inst.debug_name == kPlaceholder);
    auto offset = relativeJump(from_inst, to_inst);
    inst.callback = [offset](Stack & stack) {
      auto t = tensor_as<int64_t>(pop(stack));
      return (t == 0) ? offset : 0;
    };
    inst.debug_name = kJumpZ;
  }

  // jump when input is not 0
  void createJumpNZ(int from_inst, int to_inst) {
    auto & inst = instructions[from_inst];
    JIT_ASSERT(inst.debug_name == kPlaceholder);
    auto offset = relativeJump(from_inst, to_inst);
    inst.callback = [offset](Stack & stack) {
      auto t = tensor_as<int64_t>(pop(stack));
      return (t != 0) ? offset : 0;
    };
    inst.debug_name = kJumpNZ;
  }

  void createJump(int from_inst, int to_inst) {
    auto & inst = instructions[from_inst];
    JIT_ASSERT(inst.debug_name == kPlaceholder);
    auto offset = relativeJump(from_inst, to_inst);
    inst.callback = [=](Stack & stack) {
      return offset;
    };
    inst.debug_name = kJump;
  }

  void insertNodesFromBlock(Block* block) {
    for(auto node : block->nodes()) {
      switch(node->kind()) {
        case kIf: {
          // x = if c:
          //   <then_block>
          //   -> (vt)
          // else:
          //    <else_block>
          //   -> (vf)

          // turns into:
          //   JumpNZ c, then
          //   <else_block>
          //   x = vf
          //   Jump end
          // then:
          //   <then_block>
          //   x = vt
          // end:

          // kPlaceholder instructions are replaced with branch instructions
          // when the branch target locations are known
          auto cond_branch = insertInstruction(kPlaceholder, node->inputs(), moveFlags(node), {});
          auto then_block = node->blocks()[0];
          auto else_block = node->blocks()[1];
          insertNodesFromBlock(else_block);
          insertAssign(else_block->outputs(), moveFlags(else_block), node->outputs());
          auto jump = insertInstruction(kPlaceholder, {}, {}, {});
          auto then_block_start = instructions.size();
          insertNodesFromBlock(then_block);
          insertAssign(then_block->outputs(), moveFlags(then_block), node->outputs());
          createJump(jump, instructions.size());
          createJumpNZ(cond_branch, then_block_start);
        } break;
        case kLoop: {
          // o0 = while c i0
          //        block 0: l0
          //          <body>
          //          -> (v0, v1)

          // turns into:
          // l0 = i0
          // JumpZ c, end
          // begin:
          //   <body>
          //   c, l0 = v0, v1
          //   JumpNZ c, begin
          // end:

          auto body_block = node->blocks()[0];

          // before assign op: stack: ... <cond> <loop-carried-depdencies>
          insertAssign(node->inputs(), moveFlags(node), body_block->inputs());
          // after assign op: stack: ... <cond>
          // cond_branch consumes <cond> from top of the stack
          auto cond_branch = insertInstruction(kPlaceholder, {}, {}, {});
          // after branch: stack: ...

          auto entry = instructions.size();
          insertNodesFromBlock(body_block);
          // before assign op: stack: ... <cond> <loop-carried-depdencies>
          insertAssign(body_block->outputs(), moveFlags(body_block), body_block->inputs());
          // after assign op: stack: ... <cond>
          auto cond_branch_end = insertInstruction(kPlaceholder, {}, {}, {});
          // after branch: stack: ...

          aliasRegistersTo(node->outputs(), body_block->inputs());
          createJumpZ(cond_branch, instructions.size());
          createJumpNZ(cond_branch_end, entry);
        } break;
        default: {
          insertInstruction(node);
        } break;
      }
      // each stage ends with a load instruction
      // we record where these instructions occur, and use them to
      // exit the interpreter
      if(node->kind() == kLoad) {
        stage_end.push_back(instructions.size());
      }
    }
  }

  size_t insertInstruction(Node * n) {
    auto inst = insertInstruction(n->kind(), n->inputs(), moveFlags(n) , n->outputs());
    instructions[inst].callback = getOperation(n, computes_on_variables);
    return inst;
  }
  size_t insertInstruction(Symbol sym,
                                 ArrayRef<Value*> inputs,
                                 ArrayRef<uint8_t> move_flags,
                                 ArrayRef<Value*> outputs) {
    instructions.emplace_back();
    auto & inst = instructions.back();
    inst.debug_name = sym;
    listBegin(inst.inputs.values);
    listBegin(inst.inputs.free_flags);
    auto free_it = move_flags.begin();
    for(auto input : inputs) {
      listInsert(inst.inputs.values, getOrAllocateRegister(input, true));
      listInsert(inst.inputs.free_flags, *free_it++);
    }
    listBegin(inst.outputs);
    for(auto output : outputs) {
      listInsert(inst.outputs, getOrAllocateRegister(output));
    }
    return instructions.size() - 1;
  }
  ArrayRef<uint8_t> moveFlags(Node * n) {
    return preprocess.move_flags.at(n);
  }
  ArrayRef<uint8_t> moveFlags(Block *b) {
    return moveFlags(b->return_node());
  }

  size_t insertAssign(ArrayRef<Value*> inputs, ArrayRef<uint8_t> move_flags, ArrayRef<Value*> outputs) {
    auto inst = insertInstruction(kAssign, inputs, move_flags, outputs);
    // like Load/Store Assign does nothing since all actions are encoded
    // in inputs/outputs
    instructions[inst].callback = [](Stack& stack) { return 0; };
    return inst;
  }

  // helpers to build/access RegList objects
  int get(const ListHandle<int> & list, int i)  const {
    return int_data[list.start + i];
  }
  bool get(const ListHandle<bool> & list, int i) const {
    return bool_data[list.start + i];
  }
  void listBegin(ListHandle<int> & list) {
    list.start = int_data.size();
    list.size = 0;
  }
  void listInsert(ListHandle<int> & list, int value) {
    JIT_ASSERTM(list.start + list.size == (int)int_data.size(), "another list already started");
    int_data.push_back(value);
    list.size++;
  }
  void listBegin(ListHandle<bool> & list) {
    list.start = bool_data.size();
    list.size = 0;
  }
  void listInsert(ListHandle<bool> & list, int value) {
    JIT_ASSERTM(list.start + list.size == (int)bool_data.size(), "another list already started");
    bool_data.push_back(value);
    list.size++;
  }

  void aliasRegistersTo(ArrayRef<Value*> new_allocations, ArrayRef<Value*> existing_allocations) {
    JIT_ASSERT(new_allocations.size() == existing_allocations.size());
    for(size_t i = 0; i < new_allocations.size(); ++i) {
      auto n = new_allocations[i]->unique();
      auto e = existing_allocations[i]->unique();
      JIT_ASSERT(unique_to_reg.count(e) > 0 && unique_to_reg.count(n) == 0);
      unique_to_reg[n] = unique_to_reg[e];
    }
  }
  int getOrAllocateRegister(Value * n, bool required = false) {
    size_t u = n->unique();
    if(unique_to_reg.count(u) > 0)
      return unique_to_reg[u];
    JIT_ASSERT(!required);
    int r = register_size++;
    unique_to_reg[u] = r;
    return r;
  }

  void dumpInstruction(std::ostream & out, size_t pc) const {
    auto writeList = [&](const ListHandle<int> & list) {
      for(int i = 0; i < list.size; i++) {
        if(i > 0)
          out << ", ";
        out << get(list, i);
      }
    };
    auto writeUseList = [&](const UseList & list) {
      for(int i = 0; i < list.values.size; i++) {
        if(i > 0)
          out << ", ";
        if(get(list.free_flags, i))
          out << "move(" << get(list.values, i) << ")";
        else
          out << get(list.values, i);
      }
    };
    auto & inst = instructions.at(pc);
    writeList(inst.outputs);
    out << " = " << inst.debug_name.toString() << " ";
    writeUseList(inst.inputs);
  }
  void dump(std::ostream & out) const {
    for(size_t i = 0; i < instructions.size(); ++i) {
      dumpInstruction(out, i);
      out << "\n";
    }
  }

  // We MUST hold onto graph here because some Operators stored in the
  // instruction lists have dependencies on meta-data stored in the graph
  // that would be dead otherwise.
  // It is also very useful for debugging interpreter problems to
  // keep this around.
  std::shared_ptr<Graph> graph;
  bool computes_on_variables;
  PreprocessGraph preprocess;

  std::unordered_map<size_t, int> unique_to_reg; // map from unique of nodes to register in register table

  friend struct InterpreterState;
  std::vector<Instruction> instructions;
  std::vector<size_t> stage_end; // each stage runs while(pc < stage_end[stage])
  int register_size = 0;

  // all memory ArrayRef<int> are slices of this, to make sure
  // the interpreter is mostly linearly scanning through memory
  std::vector<int> int_data;
  std::vector<bool> bool_data;
};

// InterpreterState state that is held across stages and used to compute a Code
struct InterpreterStateImpl {
  InterpreterStateImpl(const Code & function_)
  : function(function_.pImpl),
    int_data(function->int_data.data()),
    bool_data(function->bool_data),
    registers(function->register_size) {
  }
  void runOneStage(Stack & stack) {
    // std::cout << "running stage: " << current_stage << " of " << function->stage_end.size() << "\n";
    // std::cout << *function->graph << "\n";
    // function->dump(std::cout);
    size_t pc = current_pc;
    size_t last = function->stage_end[current_stage];
    auto & instructions = function->instructions;
    while(pc < last) {
        // std::cout << "executing " << pc << ": ";
        // function->dumpInstruction(std::cout, pc);
        // std::cout << "\n";
        auto & inst = instructions[pc];
        loadTensorsFromRegisters(inst.inputs, stack);
        pc += 1 + inst.callback(stack);
        for(int i = inst.outputs.size - 1; i >= 0; i--) {
          int reg = get(inst.outputs,i);
          registers[reg] = pop(stack);
          // std::cout << "pop reg[" << reg << "];\n" << registers[reg].pImpl << "\n";
        }
    }
    current_pc = pc;
    current_stage++;
  }
  const TensorType & tensorTypeForInput(size_t i) const {
    return *function->preprocess.stage_input_types.at(current_stage).at(i)->expect<TensorType>();
  }
  int get(const ListHandle<int> & list, int i) {
    return int_data[list.start + i];
  };
  bool get(const ListHandle<bool> & list, int i) {
    return bool_data[list.start + i];
  }
  void loadTensorsFromRegisters(const UseList & uses, Stack & stack) {
    for(int i = 0; i < uses.values.size; i++) {
      int reg = get(uses.values,i);
      // std::cout << "push reg[" << reg << "];\n" << registers[reg] << "\n\n";
      if(get(uses.free_flags,i)) {
        stack.push_back(std::move(registers[reg]));
      } else {
        stack.push_back(registers[reg]);
      }

    }
  }
  size_t current_stage = 0;
  size_t current_pc = 0;
  std::shared_ptr<CodeImpl> function; // keep function alive
  // these are just copies of function to prevent indirections in interpreter
  int * int_data;
  const std::vector<bool> & bool_data;


  // this holds all the tensors for this interpreter run
  // we don't bother minimizing the size of this vector, since the extra
  // memory used by the pointers in this will be small
  // instead we are very aggresive about releasing tensors when they become dead
  // to make sure memory management happens efficiently.

  // We optimize for the case where derivatives are run with retain_graph=False
  // in the case where it is true, then the interpreter and this array get copied
  // if this every becomes a bottleneck then we _should_ consider minimizing the
  // total number or register
  std::vector<at::Tensor> registers;

  // single buffer for input/output calls to ATen functions, so that we do not reallocate
  Stack stack;
};

std::ostream & operator<<(std::ostream & out, const Code & code) {
  out << *code.pImpl->graph << "\n";
  code.pImpl->dump(out);
  return out;
}

Code::Code(std::shared_ptr<Graph> & graph, bool computes_on_variables)
: pImpl(new CodeImpl(graph, computes_on_variables)) {}
Code::~Code() {}
InterpreterState::InterpreterState(const Code & function)
: pImpl(new InterpreterStateImpl(function)) {}
InterpreterState::~InterpreterState() {}
void InterpreterState::runOneStage(Stack & stack) {
    return pImpl->runOneStage(stack);
}
const TensorType & InterpreterState::tensorTypeForInput(size_t i) const {
  return pImpl->tensorTypeForInput(i);
}
InterpreterState InterpreterState::clone() const {
  return InterpreterState(new InterpreterStateImpl(*pImpl));
}
InterpreterState::InterpreterState(InterpreterStateImpl * pImpl) : pImpl(pImpl) {}

}}
