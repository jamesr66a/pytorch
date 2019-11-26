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
          } break;
          case at::kFloat: {
            n->blocks().at(0)->inputs()[i]->setType(FloatType::get());
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
void FuseElementWise(std::shared_ptr<Graph>& graph) {}

// Expand loop nest

void ExpandLoopNest(std::shared_ptr<Graph>& graph) {
  for (auto n_itr = graph->nodes().begin(); n_itr != graph->nodes().end();) {
    Node* n = *n_itr++;
    if (n->kind() == at::loop::ElementWise) {
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
  ExpandLoopNest(lowered_graph);

  std::cout << *lowered_graph << "\n";
}

} // namespace passes
} // namespace jit
} // namespace torch
