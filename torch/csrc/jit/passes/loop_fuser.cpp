#include <torch/csrc/jit/passes/loop_fuser.h>

#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {
namespace passes {

namespace {

// Lowerings
const char* add_lowering = R"(
graph(%self : Tensor, %other : Tensor, %alpha : Scalar):
  %result = loop::ElementWise(%self, %other)
    block0(%self_val : Scalar, %other_val : Scalar):
      %premul : Scalar = scalar::mul(%alpha, %other_val)
      %result_val : Scalar = scalar::add(%self_val, %premul)
      -> (%result_val)
  return (%result) )";

static std::unique_ptr<Graph> add_lowering_graph = []() {
  std::unique_ptr<Graph> retval(new Graph());
  script::parseIR(add_lowering, retval.get());
  return retval;
}();

const char* mul_lowering = R"(
graph(%self : Tensor, %other : Tensor):
  %result = loop::ElementWise(%self, %other)
    block0(%self_val : Scalar, %other_val : Scalar):
      %result_val : Scalar = scalar::mul(%self_val, %other_val)
      -> (%result_val)
  return (%result) )";

static std::unique_ptr<Graph> mul_lowering_graph = []() {
  std::unique_ptr<Graph> retval(new Graph());
  script::parseIR(mul_lowering, retval.get());
  return retval;
}();

void LowerNodes(std::shared_ptr<Graph>& graph) {
  std::unordered_map<Value*, Value*> map;
  for (auto n_itr = graph->nodes().begin(); n_itr != graph->nodes().end();) {
    Node* n = *n_itr++;

    WithInsertPoint guard(n);
    switch (n->kind()) {
      case aten::add: {
        auto vals = insertGraph(*graph, *add_lowering_graph, n->inputs(), map);
        TORCH_INTERNAL_ASSERT(vals.size() == n->outputs().size());
        for (size_t i = 0; i < vals.size(); ++i) {
          vals[i]->copyMetadata(n->output(i));
          n->output(i)->replaceAllUsesWith(vals[i]);
        }
        n->destroy();
      } break;
      case aten::mul: {
        auto vals = insertGraph(*graph, *mul_lowering_graph, n->inputs(), map);
        TORCH_INTERNAL_ASSERT(vals.size() == n->outputs().size());
        for (size_t i = 0; i < vals.size(); ++i) {
          vals[i]->copyMetadata(n->output(i));
          n->output(i)->replaceAllUsesWith(vals[i]);
        }
        n->destroy();
      } break;
      case prim::Constant: {
        // passthrough
      } break;
      default: {
        throw script::ErrorReport(n->sourceRange())
            << "Unsupported node kind " << n->kind().toDisplayString();
      }
    }
  }
}

// Scalar type propagation

void PropagateAndResolvePromotion(Node* elementwise_node) {
  Block* b = elementwise_node->blocks()[0];
  for (auto node_itr = b->nodes().begin(); node_itr != b->nodes().end();) {
    Node* n = *node_itr++;
    switch (n->kind()) {
      case at::scalar::add:
      case at::scalar::mul: {
        enum class PromotionLattice : int32_t {
          NONE = 0,
          INT = 1,
          FLOAT = 2,
        } lattice = PromotionLattice::NONE;
        for (Value* i : n->inputs()) {
          if (i->type() == IntType::get() && lattice < PromotionLattice::INT) {
            lattice = PromotionLattice::INT;
          } else if (
              i->type() == FloatType::get() &&
              lattice < PromotionLattice::FLOAT) {
            lattice = PromotionLattice::FLOAT;
          }
        }

        TORCH_INTERNAL_ASSERT(lattice != PromotionLattice::NONE);

        for (size_t i = 0; i < n->inputs().size(); ++i) {
          Value* input = n->input(i);
          if (input->type() == IntType::get() &&
              lattice == PromotionLattice::FLOAT) {
            Node* cast_node = n->owningGraph()
                                  ->create(at::scalar::_float)
                                  ->setSourceRange(n->sourceRange());
            cast_node->addInput(input);
            cast_node->output()->setType(FloatType::get());
            cast_node->insertBefore(n);
            n->replaceInput(i, cast_node->output());
          }
        }

        switch (lattice) {
          case PromotionLattice::NONE: {
            TORCH_INTERNAL_ASSERT(false);
          } break;
          case PromotionLattice::INT: {
            n->output()->setType(IntType::get());
          } break;
          case PromotionLattice::FLOAT: {
            n->output()->setType(FloatType::get());
          } break;
        }
      } break;
      default: {
        throw script::ErrorReport(elementwise_node->sourceRange())
            << "Encountered unknown scalar op " << n->kind().toDisplayString();
      }
    }
  }
}

void PropagateScalarTypes(std::shared_ptr<Graph>& graph) {
  for (Node* n : graph->nodes()) {
    if (n->kind() == at::loop::ElementWise) {
      TORCH_INTERNAL_ASSERT(
          n->inputs().size() == n->blocks().at(0)->inputs().size());
      for (size_t i = 0; i < n->inputs().size(); ++i) {
        Value* input = n->input(i);
        auto tt = input->type()->cast<TensorType>();
        auto scalar_type = tt->scalarType().value();
        switch (scalar_type) {
          case at::kInt: {
            n->blocks().at(0)->inputs()[i]->setType(IntType::get());
            if (input->hasDebugName()) {
              n->blocks().at(0)->inputs()[i]->setDebugName(
                  input->debugName() + "_scalar");
            } else {
              n->blocks().at(0)->inputs()[i]->setDebugName(
                  std::string("_") + input->debugName() + "_scalar");
            }
          } break;
          case at::kFloat: {
            n->blocks().at(0)->inputs()[i]->setType(FloatType::get());
            if (input->hasDebugName()) {
              n->blocks().at(0)->inputs()[i]->setDebugName(
                  input->debugName() + "_scalar");
            } else {
              n->blocks().at(0)->inputs()[i]->setDebugName(
                  std::string("_") + input->debugName() + "_scalar");
            }
          } break;
          default: {
            throw script::ErrorReport(n->sourceRange())
                << "Unsupported scalar type " << toString(scalar_type);
          }
        } // switch (scalar_type)
      } // for (size_t i = 0; i < n->inputs().size(); ++i)
      PropagateAndResolvePromotion(n);
    } // if (n->kind() == loop::ElementWise)
  } // for (Node* n : graph->nodes())
}

c10::optional<double> get_numeric_value(Value* v) {
  if (v->node()->kind() == prim::Constant) {
    switch (v->node()->kindOf(at::attr::value)) {
      case jit::AttributeKind::f: {
        return v->node()->f(at::attr::value);
      } break;
      case jit::AttributeKind::i: {
        return v->node()->i(at::attr::value);
      } break;
      default:
        break;
    }
  }
  return c10::nullopt;
}

// returns index of input with identity value, c10::nullopt if none
c10::optional<size_t> probeIdentityValue(Node* n, double ident_value) {
  c10::optional<size_t> idx = c10::nullopt;
  for (size_t i = 0; i < n->inputs().size(); ++i) {
    Value* inp = n->input(i);
    if (auto val = get_numeric_value(inp)) {
      if (*val == ident_value) {
        idx = i;
      }
    }
  }
  return idx;
}

// Some optimization
void EliminateIdentity(std::shared_ptr<Graph>& graph) {
  for (Node* n : graph->nodes()) {
    if (n->kind() == at::loop::ElementWise) {
      Block* b = n->blocks()[0];
      for (auto sub_n_itr = b->nodes().begin();
           sub_n_itr != b->nodes().end();) {
        Node* sub_n = *sub_n_itr++;
        switch (sub_n->kind()) {
          case at::scalar::add: {
            if (auto zero_idx = probeIdentityValue(sub_n, 0.0)) {
              sub_n->output()->replaceAllUsesWith(
                  zero_idx.value() ? sub_n->input(0) : sub_n->input(1));
              sub_n->destroy();
            }
          } break;
          case at::scalar::mul: {
            if (auto zero_idx = probeIdentityValue(sub_n, 1.0)) {
              sub_n->output()->replaceAllUsesWith(
                  zero_idx.value() ? sub_n->input(0) : sub_n->input(1));
              sub_n->destroy();
            }
          } break;
        }
      }
    }
  }
}

// Fusion

at::optional<Node*> tryFuseProducer(Node* consumer, Value* producer) {
  if (producer->node()->kind() != at::loop::ElementWise) {
    return c10::nullopt;
  }

  Node* producer_node = producer->node();

  std::unordered_map<Value*, size_t> consumer_inputs;
  for (size_t i = 0; i < consumer->inputs().size(); ++i) {
    consumer_inputs[consumer->input(i)] = i;
  }

  Block* producer_block = producer_node->blocks()[0];
  Block* consumer_block = consumer->blocks()[0];

  std::unordered_map<Value*, Value*> producer_val_to_consumer_val;
  // First, transfer the inputs of the producer over to the consumer
  for (size_t i = 0; i < producer_node->inputs().size(); ++i) {
    Value* producer_inp = producer_node->input(i);
    if (consumer_inputs.count(producer_inp)) {
      size_t idx = consumer_inputs[producer_inp];
      producer_val_to_consumer_val[producer_block->inputs()[i]] =
          consumer_block->inputs()[idx];
    } else {
      consumer->addInput(producer_inp);
      Value* producer_block_inp = producer_node->blocks()[0]->inputs()[i];
      Value* new_input =
          consumer_block->addInput()->copyMetadata(producer_block_inp);
      producer_val_to_consumer_val[producer_block->inputs()[i]] = new_input;
    }
  }

  {
    WithInsertPoint guard(consumer->blocks()[0]->nodes().front());
    // Transfer all nodes and remap referenced Values
    for (Node* prod_sub_node : producer_node->blocks()[0]->nodes()) {
      Node* n = producer_node->owningGraph()->createClone(
          prod_sub_node, [&](Value* k) -> Value* {
            if (producer_val_to_consumer_val.count(k)) {
              return producer_val_to_consumer_val.at(k);
            } else {
              return k;
            }
          });
      n->owningGraph()->insertNode(n);
      auto new_outputs = n->outputs();
      auto orig_outputs = prod_sub_node->outputs();
      for (size_t i = 0; i < new_outputs.size(); ++i) {
        producer_val_to_consumer_val[orig_outputs[i]] = new_outputs[i];
      }
    }
  }

  // Transfer outputs to the consumer
  for (size_t i = 0; i < producer_node->outputs().size(); ++i) {
    Value* node_output = producer_node->output(i);
    Value* block_output = producer_node->blocks()[0]->outputs()[i];
    Value* new_block_output = producer_val_to_consumer_val.at(block_output);

    std::vector<size_t> producer_value_indices;
    for (size_t ii = 0; ii < consumer->inputs().size(); ++ii) {
      if (consumer->input(ii) == node_output) {
        producer_value_indices.push_back(ii);
      }
    }
    for (auto riter = producer_value_indices.rbegin();
         riter != producer_value_indices.rend();
         riter++) {
      auto idx = *riter;
      consumer->blocks()[0]->inputs()[idx]->replaceAllUsesWith(
          new_block_output);
      consumer->blocks()[0]->eraseInput(idx);
      consumer->removeInput(idx);
    }

    if (node_output->uses().size()) {
      consumer->blocks()[0]->registerOutput(new_block_output);
      node_output->replaceAllUsesWith(
          consumer->addOutput()->copyMetadata(node_output));
    }
  }

  producer_node->destroy();

  return consumer;
}

value_list sortReverseTopological(ArrayRef<Value*> inputs, Block* nodes_block) {
  value_list result;
  for (auto i : inputs) {
    if (i->node()->owningBlock() == nodes_block) {
      result.push_back(i);
    }
  }
  // Sort in reverse topological order
  std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
    return a->node()->isAfter(b->node());
  });
  return result;
}

std::tuple<graph_node_list::iterator, bool> tryFuseProducers(Node* consumer) {
  if (consumer->kind() == at::loop::ElementWise) {
    auto inputs =
        sortReverseTopological(consumer->inputs(), consumer->owningBlock());
    for (Value* producer : inputs) {
      auto fusion_group = tryFuseProducer(consumer, producer);
      if (fusion_group) {
        return {fusion_group.value()->reverseIterator(), true};
      }
    }
    return {++consumer->reverseIterator(), false};
  } else {
    return {++consumer->reverseIterator(), false};
  }
}

void FuseElementWise(std::shared_ptr<Graph>& graph) {
  bool any_changed = true;
  while (any_changed) {
    any_changed = false;
    for (auto it = graph->nodes().rbegin(); it != graph->nodes().rend();) {
      bool changed;
      std::tie(it, changed) = tryFuseProducers(*it);
      any_changed |= changed;
    }
  };
}

// here be spaghetti
void PrebroadcastInputs(std::shared_ptr<Graph>& graph) {
  for (Node* n : graph->nodes()) {
    if (n->kind() == at::loop::ElementWise) {
      // Figure out fully-specified broadcasted dimensions
      std::vector<int64_t> rev_dims;

      for (Value* i : n->inputs()) {
        auto tensor_type = i->type()->cast<TensorType>();
        TORCH_CHECK(tensor_type);
        auto varying_sizes = tensor_type->sizes();
        TORCH_CHECK(varying_sizes.isComplete());
        auto sizes = varying_sizes.concrete_sizes().value();
        size_t idx = 0;
        for (auto size_riter = sizes.rbegin(); size_riter != sizes.rend();
             size_riter++, idx++) {
          if (rev_dims.size() <= idx) {
            rev_dims.push_back(*size_riter);
          } else {
            if (rev_dims[idx] == 1) {
              rev_dims[idx] = *size_riter;
            }
            if (*size_riter == 1) {
              continue;
            }
            TORCH_CHECK(rev_dims[idx] == *size_riter);
          } // if (rev_dims.size() < idx || rev_dims[idx] == 1)
        } // for (auto size_riter = sizes.rbegin(); size_riter != sizes.rend();
          // size_riter++, idx++)
      } // for (Value *i : n->inputs())

      // Now broadcast out the operands
      WithInsertPoint guard(n);

      std::vector<int64_t> forward_dims(rev_dims);
      std::reverse(forward_dims.begin(), forward_dims.end());

      for (size_t i = 0; i < n->inputs().size(); ++i) {
        Value* inp = n->input(i);

        auto local_sizes =
            inp->type()->cast<TensorType>()->sizes().concrete_sizes().value();
        auto local_strides =
            inp->type()->cast<TensorType>()->strides().concrete_sizes().value();
        std::reverse(local_sizes.begin(), local_sizes.end());
        std::reverse(local_strides.begin(), local_strides.end());

        std::vector<int64_t> local_broadcast_dims;
        std::vector<int64_t> new_stride;
        for (size_t i = 0; i < rev_dims.size(); ++i) {
          int64_t fwd_idx = rev_dims.size() - i - 1;
          if (local_sizes.size() <= i || local_sizes.at(i) != rev_dims.at(i)) {
            local_broadcast_dims.insert(local_broadcast_dims.begin(), fwd_idx);
            new_stride.insert(new_stride.begin(), 0);
          } else {
            new_stride.insert(new_stride.begin(), local_strides.at(i));
          }
        }

        if (local_broadcast_dims.size()) {
          Node* expand_node = graph->create(at::loop::_preexpand);
          expand_node->addInput(inp);
          expand_node->is_(at::attr::dims, local_broadcast_dims);
          expand_node->output()->setType(
              inp->type()->cast<TensorType>()->withSizesStrides(
                  forward_dims, new_stride));
          Value* broadcasted = graph->insertNode(expand_node)->output();
          n->replaceInput(i, broadcasted);
        }
      }

      n->is_(at::attr::sizes, forward_dims);
    } // if (n->kind() == at::loop::ElementWise)
  } // for (Node *n : graph->nodes())
}

struct LoopNestInfo {
  c10::optional<std::unordered_map<Value*, Value*>> indexes;
  std::unordered_map<Value*, Value*> outputs;
  Block* emit_block;
};

void ExpandElementwiseLoopNestHelper(
    Node* n,
    LoopNestInfo info,
    size_t nest_level) {
  Graph* g = n->owningGraph();

  WithInsertPoint guard(info.emit_block);

  std::unordered_map<Value*, Value*> next_level_indexes;

  // Insert indexing computation for inputs and outputs
  std::vector<Value*> values_to_index(n->inputs().begin(), n->inputs().end());
  values_to_index.insert(
      values_to_index.end(), n->outputs().begin(), n->outputs().end());
  for (Value* val : values_to_index) {
    const auto strides =
        val->type()->cast<TensorType>()->strides().concrete_sizes().value();
    int64_t this_stride = strides[nest_level];
    Value* stride_value =
        g->insertConstant(this_stride)
            ->setDebugName(
                val->debugName() + "_stride_" + std::to_string(nest_level));
    Node* stride_mul = g->insertNode(g->create(at::scalar::mul));
    stride_mul->addInput(info.emit_block->inputs()[0]);
    stride_mul->addInput(stride_value);
    stride_mul->output()->setType(IntType::get());

    Value* this_index;
    if (info.indexes) {
      Node* add = g->insertNode(g->create(at::scalar::add));
      add->addInput(info.indexes.value()[val]);
      add->addInput(stride_mul->output());
      add->output()->setType(IntType::get());
      this_index = add->output();
    } else {
      this_index = stride_mul->output();
    }

    this_index->setDebugName(
        val->debugName() + "_idx_" + std::to_string(nest_level));

    next_level_indexes[val] = this_index;
  }

  if (nest_level + 1 != n->is(at::attr::sizes).size()) {
    Node* for_range = g->insertNode(g->create(at::loop::ForRange));
    info.emit_block = for_range->addBlock();
    info.emit_block->addInput(
        std::string("i") + std::to_string(nest_level + 1));

    LoopNestInfo next_level_info{
        next_level_indexes, info.outputs, for_range->blocks()[0]};
    ExpandElementwiseLoopNestHelper(n, next_level_info, nest_level + 1);
  } else {
    // Index into each tensor

    // Map from original block input values to the new indexed values.
    // We use the original block inputs as a key so we can resolve new
    // values when we clone the nodes.
    std::unordered_map<Value*, Value*> remapped_values;

    TORCH_INTERNAL_ASSERT(
        n->inputs().size() == n->blocks()[0]->inputs().size());
    for (size_t i = 0; i < n->inputs().size(); ++i) {
      Value* node_input = n->input(i);
      Value* block_input = n->blocks()[0]->inputs()[i];

      Node* read_node = g->insertNode(g->create(at::scalar::LinearIndexedRead));
      read_node->addInput(node_input);
      read_node->addInput(next_level_indexes.at(node_input));
      read_node->output()->copyMetadata(block_input);
      remapped_values[block_input] = read_node->output();
    }

    std::unordered_map<Value*, Value*> orig_block_outputs_to_node_outputs;
    TORCH_INTERNAL_ASSERT(
        n->outputs().size() == n->blocks()[0]->outputs().size());
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      orig_block_outputs_to_node_outputs[n->blocks()[0]->outputs()[i]] =
          n->outputs()[i];
    }

    std::unordered_map<Value*, Value*> new_outputs;
    for (Node* sub_node : n->blocks()[0]->nodes()) {
      Node* new_node =
          g->insertNode(g->createClone(sub_node, [&](Value* v) -> Value* {
            if (remapped_values.count(v)) {
              return remapped_values[v];
            } else {
              return v;
            }
          }));
      for (size_t i = 0; i < sub_node->outputs().size(); i++) {
        remapped_values[sub_node->output(i)] = new_node->output(i);
      }
      if (orig_block_outputs_to_node_outputs.count(sub_node->output())) {
        new_outputs[sub_node->output()] = new_node->output();
      }
    }

    for (const auto& kv : new_outputs) {
      Value* orig_node_output = orig_block_outputs_to_node_outputs.at(kv.first);
      Node* write_node =
          g->insertNode(g->create(at::scalar::LinearIndexedWrite, 0));
      write_node->addInput(info.outputs.at(orig_node_output));
      write_node->addInput(next_level_indexes.at(orig_node_output));
      write_node->addInput(kv.second);
    }
  }
}

// Stages:
// 1) Create alloc nodes for each output with the proper size
// 2) Recursively, for each iterated dimension:
//    3) emit loop node over the size of that output dimension
//    4) For each iterated value and output, emit indexing computation for that
//    dimension 5) For the innermost loop:
//        6) emit indexing expression to get concrete scalar value
//        7) Emit loop body, mapping original block inputs to the indexed scalar
//        vals 8) Emit output write expressions
void ExpandLoopNest(std::shared_ptr<Graph>& graph) {
  for (auto n_itr = graph->nodes().begin(); n_itr != graph->nodes().end();) {
    Node* n = *n_itr++;
    if (n->kind() == at::loop::ElementWise) {
      WithInsertPoint guard(n);

      Graph* g = n->owningGraph();
      LoopNestInfo info;
      // Alloc outputs
      for (size_t i = 0; i < n->outputs().size(); ++i) {
        Value* n_output = n->outputs()[i];
        Value* alloced_output =
            g->insertNode(g->create(at::loop::Alloc))->output();
        alloced_output->copyMetadata(n_output);
        n_output->replaceAllUsesWith(alloced_output);
        info.outputs[n_output] = alloced_output;
      }
      Node* outer_for_range = g->insertNode(g->create(at::loop::ForRange));
      info.emit_block = outer_for_range->addBlock();
      info.emit_block->addInput("i");
      ExpandElementwiseLoopNestHelper(n, info, 0);

      n->destroy();
    }
  }
}

} // namespace

void LoopFuser(const std::shared_ptr<Graph>& graph) {
  auto lowered_graph = graph->copy();

  std::cout << *graph << std::endl;

  LowerNodes(lowered_graph);
  EliminateIdentity(lowered_graph);
  PropagateScalarTypes(lowered_graph);
  FuseElementWise(lowered_graph);
  PrebroadcastInputs(lowered_graph);
  ExpandLoopNest(lowered_graph);

  std::cout << *lowered_graph << "\n";
}

} // namespace passes
} // namespace jit
} // namespace torch
