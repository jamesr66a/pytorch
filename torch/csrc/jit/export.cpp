#include "torch/csrc/jit/export.h"
#include "torch/csrc/autograd/symbolic.h"
#include "onnx/onnx/onnx.pb.h"
#include "torch/csrc/onnx/onnx.h"

#include "torch/csrc/utils/functional.h"
#include <ATen/ATen.h>
#include <ATen/optional.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

namespace torch { namespace jit {

namespace {

namespace onnx = ::torch::onnx;

std::string value_name(Value* n) {
  return n->uniqueName();
}

struct ExportContext {
  size_t num_blocks = 0;
  onnx::OperatorExportTypes operator_export_type;
};

void encodeGraph(onnx_torch::GraphProto * p_g, const std::shared_ptr<Graph> & g,
                 const std::vector<at::Tensor> & initializers,
                 ExportContext *ctx, RawDataExportMap* raw_data_export_map=nullptr);

void encodeBlock(onnx_torch::GraphProto * p_g, Block *b,
                const std::vector<at::Tensor> & initializers,
                ExportContext *ctx, RawDataExportMap* raw_data_export_map);

void encodeTensor(onnx_torch::TensorProto * p, const at::Tensor & tensor,
                  at::optional<std::string> external_ref={},
                  RawDataExportMap* raw_data_export_map = nullptr) {
  for(auto d : tensor.sizes()) {
    p->add_dims(d);
  }
  onnx_torch::TensorProto_DataType onnx_type;
  // Most integral types and float16 need to be serialized as int32
  at::ScalarType cast_type = tensor.type().scalarType();
  switch(tensor.type().scalarType()) {
    case at::kDouble:
      onnx_type = onnx_torch::TensorProto_DataType_DOUBLE;
      break;
    case at::kFloat:
      onnx_type = onnx_torch::TensorProto_DataType_FLOAT;
      break;
    case at::kHalf:
      onnx_type = onnx_torch::TensorProto_DataType_FLOAT16;
      cast_type = at::kInt;
      break;
    case at::kByte:
      onnx_type = onnx_torch::TensorProto_DataType_UINT8;
      cast_type = at::kInt;
      break;
    case at::kChar:
      onnx_type = onnx_torch::TensorProto_DataType_INT8;
      cast_type = at::kInt;
      break;
    case at::kShort:
      onnx_type = onnx_torch::TensorProto_DataType_INT16;
      cast_type = at::kInt;
      break;
    case at::kInt:
      onnx_type = onnx_torch::TensorProto_DataType_INT32;
      break;
    case at::kLong:
      onnx_type = onnx_torch::TensorProto_DataType_INT64;
      break;
    default:
      torch::barf("unexpected tensor scalar type");
      break;
  }
  p->set_data_type(onnx_type);
  // CPU's HalfTensor doesn't have contiguous(), so first calling contiguous()
  auto t = tensor.contiguous().toBackend(at::kCPU).toType(cast_type);
  // Add a buffer to the raw_data_export_map for the caller to dump into an
  // external data store. If external_ref is not specified, we instead dump
  // the contiguous data into the protobuf itself
  if (external_ref) {
    // For now, we use the name of the tensor as the external lookup name to
    // avoid ONNX protobuf changes.
    JIT_ASSERT(external_ref.value() == p->name());
    JIT_ASSERT(raw_data_export_map != nullptr);
    JIT_ASSERT(raw_data_export_map->count(external_ref.value()) == 0);
    (*raw_data_export_map)[external_ref.value()] = t;
    p->set_raw_data("__EXTERNAL");
  } else {
    JIT_ASSERT(t.is_contiguous());
    p->set_raw_data(std::string(static_cast<char*>(t.data_ptr()),  t.type().elementSizeInBytes() * t.numel()));
  }
}

void addAttribute(onnx_torch::NodeProto * n_p, jit::Node * n, jit::Symbol name, ExportContext *ctx) {
  auto attr = n_p->add_attribute();
  JIT_ASSERT(name.is_attr());
  attr->set_name(name.toUnqualString());
  switch(n->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(n->f(name));
      attr->set_type(onnx_torch::AttributeProto_AttributeType_FLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(onnx_torch::AttributeProto_AttributeType_FLOATS);
      for(auto & v : n->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_type(onnx_torch::AttributeProto_AttributeType_INT);
      attr->set_i(n->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(onnx_torch::AttributeProto_AttributeType_INTS);
      for(auto & v : n->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(onnx_torch::AttributeProto_AttributeType_STRING);
      attr->set_s(n->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(onnx_torch::AttributeProto_AttributeType_STRINGS);
      for(auto & v : n->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(onnx_torch::AttributeProto_AttributeType_TENSOR);
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(onnx_torch::AttributeProto_AttributeType_TENSORS);
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(onnx_torch::AttributeProto_AttributeType_GRAPH);
      auto g = attr->mutable_g();
      encodeGraph(g, n->g(name), {}, ctx, nullptr);
    } break;
    case AttributeKind::gs:
      attr->set_type(onnx_torch::AttributeProto_AttributeType_GRAPHS);
      for(auto & v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v, {}, ctx, nullptr);
      }
      break;
  }
}

void encodeTypeProtoTensorType(onnx_torch::TypeProto_Tensor* tensor_type, Value* n) {
  onnx_torch::TensorShapeProto* shape = tensor_type->mutable_shape();
  if (TensorType* node_type = n->type()->cast<TensorType>()) {
    const std::vector<std::int64_t>& sizes = node_type->sizes();
    for (size_t i = 0; i < sizes.size(); i++) {
      shape->add_dim();
      shape->mutable_dim(i)->set_dim_value(sizes[i]);
    }
    onnx_torch::TensorProto_DataType onnx_type;
    switch(node_type->scalarType()) {
      case at::kDouble:
        onnx_type = onnx_torch::TensorProto_DataType_DOUBLE;
        break;
      case at::kFloat:
        onnx_type = onnx_torch::TensorProto_DataType_FLOAT;
        break;
      case at::kHalf:
        onnx_type = onnx_torch::TensorProto_DataType_FLOAT16;
        break;
      case at::kByte:
        onnx_type = onnx_torch::TensorProto_DataType_UINT8;
        break;
      case at::kChar:
        onnx_type = onnx_torch::TensorProto_DataType_INT8;
        break;
      case at::kShort:
        onnx_type = onnx_torch::TensorProto_DataType_INT16;
        break;
      case at::kInt:
        onnx_type = onnx_torch::TensorProto_DataType_INT32;
        break;
      case at::kLong:
        onnx_type = onnx_torch::TensorProto_DataType_INT64;
        break;
      default:
        torch::barf("unexpected tensor scalar type");
        break;
    }
    tensor_type->set_elem_type(onnx_type);
  }
}

void encodeValueInfo(onnx_torch::ValueInfoProto* v, Value* n) {
  v->set_name(value_name(n));
  onnx_torch::TypeProto* t = v->mutable_type();
  onnx_torch::TypeProto_Tensor* tensor_type = t->mutable_tensor_type();
  encodeTypeProtoTensorType(tensor_type, n);
}

void encodeGraph(onnx_torch::GraphProto * p_g, const std::shared_ptr<Graph>& g,
                 const std::vector<at::Tensor> & initializers,
                 ExportContext *ctx, RawDataExportMap* raw_data_export_map) {
  encodeBlock(p_g, g->block(), initializers, ctx, raw_data_export_map);
}

void encodeBlock(onnx_torch::GraphProto * p_g, Block *b,
                 const std::vector<at::Tensor> & initializers,
                 ExportContext *ctx, RawDataExportMap* raw_data_export_map) {
  JIT_ASSERT(p_g != nullptr);
  std::string block_name = "torch-jit-export";
  if (ctx->num_blocks) {
    block_name += std::to_string(ctx->num_blocks);
  }
  ctx->num_blocks++;
  p_g->set_name(block_name);

  for (auto input : b->inputs()) {
    onnx_torch::ValueInfoProto* v = p_g->add_input();
    encodeValueInfo(v, input);
  }
  for (auto output : b->outputs()) {
    onnx_torch::ValueInfoProto* v = p_g->add_output();
    encodeValueInfo(v, output);
  }
  for (auto node : b->nodes()) {
    bool is_raw_export = ctx->operator_export_type == onnx::OperatorExportTypes::RAW;
    if (node->kind() == prim::Undefined && !is_raw_export) {
      // Undefined nodes are used to implement optional inputs. One
      // way to "not provide" an optional input is to create an
      // Undefined node, and pass its output as that input.
      continue;
    }
    auto p_n = p_g->add_node();
    if (node->getSourceLocation()) {
      std::stringstream ss;
      node->getSourceLocation()->highlight(ss);
      p_n->set_doc_string(ss.str());
    }
    for(auto input : node->inputs()) {
      if (input->node()->kind() == prim::Undefined && !is_raw_export) {
        p_n->add_input("");
      } else {
        p_n->add_input(value_name(input));
      }
    }
    for(auto output : node->outputs()) {
      p_n->add_output(value_name(output));
    }
    if (is_raw_export) {
      JIT_ASSERT(!node->kind().is_onnx());
      p_n->set_domain(node->kind().domainString());
    }
    else if (ctx->operator_export_type != onnx::OperatorExportTypes::ONNX_ATEN_FALLBACK) {
      JIT_ASSERT(node->kind().is_onnx());
    }
    p_n->set_op_type(node->kind().toUnqualString());
    for(auto attr_name : node->attributeNames()) {
      addAttribute(p_n, node, attr_name, ctx);
    }
    if (is_raw_export && node->blocks().size() > 0) {
      auto blocks = p_n->add_attribute();
      blocks->set_name("_blocks");
      blocks->set_type(onnx_torch::AttributeProto_AttributeType_GRAPHS);
      for (auto block : node->blocks()) {
        auto graph = blocks->add_graphs();
        encodeBlock(graph, block, initializers, ctx, raw_data_export_map);
      }
    }
    if (node->kind() == torch::jit::onnx::Loop) {
      JIT_ASSERT(node->blocks().size() == 1);

      auto body = p_n->add_attribute();
      body->set_name("body");
      body->set_type(onnx_torch::AttributeProto_AttributeType_GRAPH);
      auto g = body->mutable_g();
      encodeBlock(g, node->blocks()[0], {}, ctx, raw_data_export_map);
    }
    if (node->kind() == torch::jit::onnx::If) {
      JIT_ASSERT(node->blocks().size() == 2);

      auto true_branch = p_n->add_attribute();
      true_branch->set_name("then_branch");
      true_branch->set_type(onnx_torch::AttributeProto_AttributeType_GRAPH);
      auto true_g = true_branch->mutable_g();
      encodeBlock(true_g, node->blocks()[0], {}, ctx, raw_data_export_map);

      auto false_branch = p_n->add_attribute();
      false_branch->set_name("else_branch");
      false_branch->set_type(onnx_torch::AttributeProto_AttributeType_GRAPH);
      auto false_g = false_branch->mutable_g();
      encodeBlock(false_g, node->blocks()[1], {}, ctx, raw_data_export_map);
    }
  }
  auto num_initializers = initializers.size();
  JIT_ASSERT(b->inputs().size() >= num_initializers);
  size_t inputs_count = b->inputs().size() - num_initializers;
  for (auto & tensor : initializers) {
    // TODO: stop using positions to determine which initializers
    // match to which inputs
    std::string name = p_g->input(inputs_count++).name();
    auto p = p_g->add_initializer();
    p->set_name(name);
    if (raw_data_export_map) {
      encodeTensor(p, tensor, name, raw_data_export_map);
    } else {
      encodeTensor(p, tensor, {});
    }
  }
}

void encodeModel(onnx_torch::ModelProto* p_m, const std::shared_ptr<Graph>& g,
                 const std::vector<at::Tensor>& initializers,
                 RawDataExportMap* raw_data_export_map = nullptr,
                 onnx::OperatorExportTypes operator_export_type
                   = onnx::OperatorExportTypes::ONNX) {
  onnx_torch::GraphProto* p_g = p_m->mutable_graph();
  ExportContext ctx;
  ctx.operator_export_type = operator_export_type;
  encodeGraph(p_g, g, initializers, &ctx, raw_data_export_map);
}

namespace {
std::string getNodeStackTraceString(Node* n) {
  std::stringstream ss;
  if (n->getSourceLocation()) {
    n->getSourceLocation()->highlight(ss);
  } else {
    ss << "<unknown location>";
  }
  return ss.str();
}
} // namespace

void validateGraph(const std::shared_ptr<Graph>& graph, onnx::OperatorExportTypes operator_export_type) {
  for (auto node : graph->nodes()) {
      // Macro'ed so we get a marginally better line number on failed export
#define FAIL_EXPORT(name) \
      throw std::runtime_error(std::string("ONNX export failed: ") + name + "\n\nGraph we tried to export:\n" + graph->toString());
    IR_IF(node, PythonOp)
      auto py_node = static_cast<torch::jit::PythonOp*>(value);
      FAIL_EXPORT(
          "Couldn't export Python operator " + py_node->name() +
          "\n\nDefined at:\n" + getNodeStackTraceString(node))
    IR_ELSE()
      // Special error messages for certain types of operators
      if (node->kind() == aten::expand) {
        FAIL_EXPORT(
            "Could not export a broadcasted operation; ONNX likely does not support this form of broadcasting.\n\nBroadcast occurred at:\n" +
            getNodeStackTraceString(node));
      }
      if (node->kind() == prim::PackPadded || node->kind() == prim::PadPacked) {
        FAIL_EXPORT(
            "Cannot export individual pack_padded_sequence or pad_packed_sequence; these operations must occur in pairs.\n\nUsage of this operation occurred at:\n" +
            getNodeStackTraceString(node));
      }
      bool is_aten_fallback = operator_export_type == onnx::OperatorExportTypes::ONNX_ATEN_FALLBACK;
      if (!node->kind().is_onnx() && !is_aten_fallback && node->kind() != prim::Undefined) {
        FAIL_EXPORT(
            "Couldn't export operator " + node->kind().toDisplayString() + "\n\nDefined at:\n" +
            getNodeStackTraceString(node));
      }
    IR_END()
#undef FAIL_EXPORT
  }
}

// Pretty printing
namespace {
constexpr char indent_char = ' ';
constexpr size_t indent_multiplier = 2;

std::string idt(size_t indent) {
  return std::string(indent * indent_multiplier, indent_char);
}

std::string nlidt(size_t indent) {
  return std::string("\n") + idt(indent);
}

void dump(onnx_torch::TensorProto, std::ostream& stream, size_t indent) {
  stream << "TensorProto shape: [";
  for (size_t i = 0; i < dims.size(); ++i) {
    stream << *dims[i] << (i == dims.size() - 1 ? "" : " ");
  }
  stream << "]";
}

void dump(onnx_torch::TensorShapeProto, std::ostream& stream, size_t indent) {
  for (size_t i=0; i < dims.size(); ++i) {
    auto &dim = dims[i];
    if (dim->has_dim_value) {
      stream << dim->dim_value;
    } else {
      stream << "?";
    }
    stream << (i == dims.size() - 1 ? "" : " ");
  }
}

void dump(onnx_torch::TypeProtoTensor, std::ostream& stream, size_t indent) {
  stream << "Tensor dims: ";
  shape->dump(stream);
}

void dump(onnx_torch::TypeProto, std::ostream& stream, size_t indent) {
  tensor_type->dump(stream);
}

void dump(onnx_torch::ValueInfoProto, std::ostream& stream, size_t indent) {
  stream << "{name: \"" << name
         << "\", type:";
  type->dump(stream);
  stream << "}";
}

void dump(onnx_torch::AttributeProto, std::ostream& stream, size_t indent) {
  stream << "{ name: '" << name << "', type: ";
  if (proto.has_f) {
    stream << "float, value: " << proto.f;
  } else if (proto.has_i) {
    stream << "int, value: " << proto.i;
  } else if (s.length()) {
    stream << "string, value: '" << s << "'";
  } else if (g) {
    stream << "graph, value:\n";
    g->dump(stream, indent+1);
    stream << nlidt(indent);
  } else if (t) {
    stream << "tensor, value:";
    t->dump(stream, indent+1);
  } else if (floats.size()) {
    stream << "floats, values: [";
    for (size_t i=0; i < floats.size(); ++i)
      stream << *floats[i] << (i == floats.size() - 1 ? "" : " ");
    stream << "]";
  } else if (ints.size()) {
    stream << "ints, values: [";
    for (size_t i=0; i < ints.size(); ++i)
      stream << *ints[i] << (i == ints.size() - 1 ? "" : " ");
    stream << "]";
  } else if (strings.size()) {
    stream << "strings, values: [";
    for (size_t i=0; i < strings.size(); ++i)
      stream << "'" << *strings[i] << "'" << (i == strings.size() - 1 ? "" : " ");
    stream << "]";
  } else if (tensors.size()) {
    stream << "tensors, values: [";
    for (auto& t : tensors) {
      t->dump(stream, indent+1);
    }
    stream << "]";
  } else if (graphs.size()) {
    stream << "graphs, values: [";
    for (auto& g : graphs) {
      g->dump(stream, indent+1);
    }
    stream << "]";
  } else {
    stream << "UNKNOWN";
  }
  stream << "}";
}

void dump(onnx_torch::NodeProto, std::ostream& stream, size_t indent) {
  stream << "Node {type: \"" << op_type << "\", inputs: [";
  for (size_t i=0; i < inputs.size(); ++i) {
    stream << *inputs[i] << (i == inputs.size() - 1 ? "" : ",");
  }
  stream << "], outputs: [";
  for (size_t i=0; i < outputs.size(); ++i) {
    stream << *outputs[i] << (i == outputs.size() - 1 ? "" : ",");
  }
  stream << "], attributes: [";
  for (size_t i=0; i < attributes.size(); ++i) {
    attributes[i]->dump(stream, indent+1);
    stream << (i == attributes.size() - 1 ? "" : ",");
  }
  stream << "]}";
}

void dump(onnx_torch::GraphProto, std::ostream& stream, size_t indent) {
  stream << idt(indent) << "GraphProto {" << nlidt(indent+1)
         << "name: \"" << name << "\"" << nlidt(indent+1)
         << "inputs: [";
  for (size_t i=0; i < inputs.size(); ++i) {
    inputs[i]->dump(stream, indent+2);
    stream << (i == inputs.size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "outputs: [";
  for (size_t i=0; i < outputs.size(); ++i) {
    outputs[i]->dump(stream, indent+2);
    stream << (i == outputs.size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "initializers: [";
  for (size_t i=0; i < initializers.size(); ++i) {
    initializers[i]->dump(stream, indent+2);
    stream << (i == initializers.size() - 1 ? "" : ",");
  }
  stream << "]" << nlidt(indent+1)
         << "nodes: [" << nlidt(indent+2);
  for (size_t i=0; i < nodes.size(); ++i) {
    nodes[i]->dump(stream, indent+2);
    if (i != nodes.size() - 1) stream << "," << nlidt(indent+2);
  }
  stream << nlidt(indent+1) << "]\n" << idt(indent) << "}\n";
}

void dump(onnx_torch::OperatorSetIdProto, std::ostream& stream, size_t indent) {
  stream << "OperatorSetIdProto { domain: " << domain << "}";
}

void dump(onnx_torch::ModelProto m_p, std::ostream& stream, size_t indent) {
  stream << idt(indent)
         << "ModelProto {" << nlidt(indent+1)
         << "producer_name: \"" << producer_name << "\"" << nlidt(indent+1)
         << "domain: \"" << domain << "\"" << nlidt(indent+1)
         << "doc_string: \"" << doc_string << "\"";
  if (m_p) {
    stream << nlidt(indent+1) << "graph:\n";
    graph->dump(graph, stream, indent+2);
  }
  if (opset_import.size()) {
    stream << idt(indent+1) << "opset_import: [";
    for (auto &opset_imp : opset_import) {
      opset_imp->dump(stream, indent+2);
    }
    stream << "],\n";
  }
  stream << idt(indent) << "}\n";
}
} // namespace

}

namespace {

RawDataExportMap ToModelProto(
    const std::shared_ptr<Graph>& graph,
    const std::vector<at::Tensor> & initializers,
    int64_t onnx_opset_version,
    bool defer_weight_export,
    onnx::OperatorExportTypes operator_export_type,
    onnx_torch::ModelProto *model_proto) {
  if (operator_export_type != onnx::OperatorExportTypes::RAW) {
    validateGraph(graph, operator_export_type);
  }

  model_proto->set_producer_name("pytorch");
  model_proto->set_producer_version("0.3");
  auto* imp = model_proto->add_opset_import();
  // This is the version of ONNX operator set we are targeting
  imp->set_version(onnx_opset_version);

  // Map {external_data_ref -> raw data} for external serialization of weights
  RawDataExportMap raw_data_export_map;

  // Set up nanopb callbacks and compute the amount of space needed to store
  // the resulting protobuf
  if (defer_weight_export) {
    encodeModel(model_proto, graph, initializers, &raw_data_export_map, operator_export_type);
  } else {
    encodeModel(model_proto, graph, initializers, nullptr, operator_export_type);
  }

  return raw_data_export_map;
}


}  // namespace


std::string PrettyPrintExportedGraph(
                        const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers,
                        int64_t onnx_opset_version,
                        bool defer_weight_export,
                        ::torch::onnx::OperatorExportTypes operator_export_type) {
  onnx_torch::ModelProto model_proto;
  RawDataExportMap raw_data_export_map;
  raw_data_export_map = ToModelProto(
    graph, initializers, onnx_opset_version, defer_weight_export, operator_export_type,
    &model_proto);
  return model_proto.DebugString();
}

// export_raw_ir will export IR ops without turning them into ONNX ops.
// The output will use the ONNX protobuf format, but the ops will not
// conform to the ONNX op specification. Thus, the output will not
// be interpretable by a ONNX-compatible framework. However, PyTorch or
// libtorch will be able to import the IR and play it back.
std::tuple<std::string, RawDataExportMap> ExportGraph(
                        const std::shared_ptr<Graph>& graph,
                        const std::vector<at::Tensor> & initializers,
                        int64_t onnx_opset_version,
                        bool defer_weight_export,
                        ::torch::onnx::OperatorExportTypes operator_export_type) {
  onnx_torch::ModelProto model_proto;
  RawDataExportMap raw_data_export_map;
  raw_data_export_map = ToModelProto(
    graph, initializers, onnx_opset_version, defer_weight_export, operator_export_type,
    &model_proto);
  return std::make_tuple(model_proto.SerializeAsString(), raw_data_export_map);
}

}}
