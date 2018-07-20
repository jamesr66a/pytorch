#include "torch/csrc/jit/import.h"
#include "onnx/onnx.pb.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/utils/functional.h"

#include <ATen/ATen.h>

#include <unordered_map>
#include <vector>
#include <string>

#include <pb_decode.h>

namespace torch { namespace jit {

namespace {

// IR graph construction

class JitDecoder {
 protected:
  std::shared_ptr<Graph> buildGraph(const onnx_torch::GraphProto& graph_proto);

  void buildBlock(const onnx_torch::GraphProto& graph_proto, Block* block,
                   std::unordered_map<std::string, Value*>& value_map);

  void buildBlocks(const std::vector<onnx_torch::GraphProto>& graphs_, Node* node,
                   std::unordered_map<std::string, Value*>& value_map);

  at::ScalarType onnxTypeToATenType(onnx_torch::TensorProto_DataType tensor_proto);

  virtual at::Tensor buildTensor(const onnx_torch::TensorProto& tensor_proto);
};

at::ScalarType JitDecoder::onnxTypeToATenType(onnx_torch::TensorProto_DataType onnx_type) {
  switch(onnx_type) {
    case onnx_torch::TensorProto_DataType_UINT8:
      return at::kByte;
    case onnx_torch::TensorProto_DataType_INT8:
      return at::kChar;
    case onnx_torch::TensorProto_DataType_INT16:
      return at::kShort;
    case onnx_torch::TensorProto_DataType_INT32:
      return at::kInt;
    case onnx_torch::TensorProto_DataType_INT64:
      return at::kLong;
    case onnx_torch::TensorProto_DataType_FLOAT16:
      return at::kHalf;
    case onnx_torch::TensorProto_DataType_FLOAT:
      return at::kFloat;
    case onnx_torch::TensorProto_DataType_DOUBLE:
      return at::kDouble;
    default:
      throw std::runtime_error("Unsupported data type");
  }
}

at::Tensor JitDecoder::buildTensor(const onnx_torch::TensorProto& tensor_proto) {
  at::Tensor tensor = at::CPU(onnxTypeToATenType(tensor_proto.data_type())).tensor();
  tensor.resize_({tensor_proto.dims().begin(), tensor_proto.dims().end()});
  TORCH_ASSERT(tensor.storage()->size() * tensor.storage()->elementSize() == tensor_proto.raw_data().size());
  std::memcpy(tensor.data_ptr(), tensor_proto.raw_data().data(), tensor_proto.raw_data().size());
  return tensor;
}

void JitDecoder::buildBlocks(
    const std::vector<onnx_torch::GraphProto>& graphs_, Node* node,
    std::unordered_map<std::string, Value*>& value_map) {
  for (auto g_ : graphs_) {
    auto block = node->addBlock();
    buildBlock(g_, block, value_map);
  }
}

std::shared_ptr<Graph> JitDecoder::buildGraph(const onnx_torch::GraphProto& graph_proto) {
  auto graph = std::make_shared<Graph>();
  std::unordered_map<std::string, Value*> value_map;

  buildBlock(graph_proto, graph->block(), value_map);

  return graph;
}

void JitDecoder::buildBlock(const onnx_torch::GraphProto& graph_proto, Block* block,
                std::unordered_map<std::string, Value*>& value_map) {

  for (auto & input : graph_proto.input()) {
    value_map[input.name()] = block->addInput();
  }

  for (auto & node_ : graph_proto.node()) {
    TORCH_ASSERT(node_.op_type() != "PythonOp");

    auto node = block->owningGraph()->create(Symbol::fromDomainAndUnqualString(node_.domain(), node_.op_type()),
                                             node_.output().size());

    for (auto & attr : node_.attribute()) {
      Symbol name = Symbol::attr(attr.name());

      switch(attr.type()) {
        case onnx_torch::AttributeProto_AttributeType_UNDEFINED:
          throw std::runtime_error("UNDEFINED attribute unsupported");
          break;
        case onnx_torch::AttributeProto_AttributeType_FLOAT:
          node->f_(name, attr.f());
          break;
        case onnx_torch::AttributeProto_AttributeType_INT:
          node->i_(name, attr.i());
          break;
        case onnx_torch::AttributeProto_AttributeType_STRING:
          node->s_(name, std::move(attr.s()));
          break;
        case onnx_torch::AttributeProto_AttributeType_TENSOR:
          node->t_(name, buildTensor(attr.t()));
          break;
        case onnx_torch::AttributeProto_AttributeType_GRAPH:
          node->g_(name, buildGraph(attr.g()));
          break;
        case onnx_torch::AttributeProto_AttributeType_FLOATS:
          node->fs_(name, {attr.floats().begin(), attr.floats().end()});
          break;
        case onnx_torch::AttributeProto_AttributeType_INTS:
          node->is_(name, {attr.ints().begin(), attr.ints().end()});
          break;
        case onnx_torch::AttributeProto_AttributeType_STRINGS:
          node->ss_(name, {attr.strings().begin(), attr.strings().end()});
          break;
        case onnx_torch::AttributeProto_AttributeType_TENSORS:
          node->ts_(name, fmap(attr.tensors(), [this](const onnx_torch::TensorProto& t) {
                                                 return buildTensor(t);
                                               }));
          break;
        case onnx_torch::AttributeProto_AttributeType_GRAPHS:
          if (attr.name() == "_blocks") {
            buildBlocks({attr.graphs().begin(), attr.graphs().end()}, node, value_map);
          }
          else {
            node->gs_(name, fmap(attr.graphs(), [this](const onnx_torch::GraphProto& g_) {
                                                  return buildGraph(g_);
                                                }));
          }
          break;
      }
    }

    for (auto & input : node_.input()) {
      auto v = value_map[input];
      node->addInput(v);
    }

    for (int i=0; i<node_.output().size(); i++) {
      value_map[node_.output(i)] = node->outputs()[i];
    }

    block->appendNode(node);
  }

  for (auto & output : graph_proto.output()) {
    Value* v = value_map.at(output.name());
    block->registerOutput(v);
  }
}

class GraphDecoder : JitDecoder {
 public:
  std::shared_ptr<Graph> decode(const std::string& serialized_graph,
                                std::vector<at::Tensor>& initializers);
};

std::shared_ptr<Graph> GraphDecoder::decode(
    const std::string& serialized_graph,
    std::vector<at::Tensor>& initializers) {
  auto model_proto = onnx_torch::ModelProto();
  model_proto.ParseFromString(serialized_graph);

  auto graph_proto = model_proto.graph();
  auto graph = buildGraph(graph_proto);
  for (auto &tensor_ : graph_proto.initializer()) {
    initializers.push_back(buildTensor(tensor_));
  }
  return graph;
}

class ModuleDecoder : JitDecoder {
 public:
  std::shared_ptr<script::Module> decode(
      std::shared_ptr<script::Module> root_module,
      const std::string& serialized_module,
      const std::unordered_map<std::string, std::string>& storage_map);

 private:
  at::Tensor buildTensor(const onnx_torch::TensorProto& tensor_proto);

  at::Tensor buildParameter(const onnx_torch::TensorProto& tensor_proto);

  std::pair<std::shared_ptr<script::Module>, std::string> parseFullName(
      std::shared_ptr<script::Module> root_module,
      const std::string fullname);

  const std::unordered_map<std::string, std::string> *storage_map_;
};

at::Tensor ModuleDecoder::buildParameter(const onnx_torch::TensorProto& tensor_proto) {
  at::Tensor tensor = buildTensor(tensor_proto);
  autograd::Variable var = autograd::make_variable(tensor, tensor_proto.int64_data(0));
  return var;
}

at::Tensor ModuleDecoder::buildTensor(const onnx_torch::TensorProto& tensor_proto) {
  at::Tensor tensor = at::CPU(onnxTypeToATenType(tensor_proto.data_type())).tensor();
  tensor.resize_({tensor_proto.dims().begin(), tensor_proto.dims().end()});
  auto it = storage_map_->find(tensor_proto.doc_string());
  JIT_ASSERT(it != storage_map_->end());
  std::memcpy(tensor.data_ptr(), it->second.data(), tensor.numel() * tensor.type().elementSizeInBytes());
  return tensor;
}

// Given a full name of a parameter or method,
// return the parent submodule and local name
std::pair<std::shared_ptr<script::Module>, std::string> ModuleDecoder::parseFullName(
    std::shared_ptr<script::Module> root_module,
    const std::string fullname) {
  std::vector<std::string> vec;
  std::stringstream ss(fullname);
  std::string name;
  while (std::getline(ss, name, '.')) {
    vec.push_back(name);
  }

  std::shared_ptr<script::Module> curr = root_module;
  for (size_t i = 0; i < vec.size() - 1; i++) {
    if (curr->find_module(vec[i]) != nullptr) {
      curr = curr->get_module(vec[i]);
    } else {
      curr->register_module(vec[i], std::make_shared<script::Module>());
      curr = curr->get_module(vec[i]);
    }
  }
  return std::make_pair(curr, vec.back());
}

std::shared_ptr<script::Module> ModuleDecoder::decode(
    const std::shared_ptr<script::Module> root_module,
    const std::string &serialized_module,
    const std::unordered_map<std::string, std::string> &storage_map) {
  auto model_proto = onnx_torch::ModelProto();
  model_proto.ParseFromString(serialized_module);
  auto graph_proto = model_proto.graph();

  std::unordered_map<std::string, at::Tensor*> param_map;
  storage_map_ = &storage_map;

  for (auto &tensor_proto : graph_proto.initializer()) {
    std::shared_ptr<script::Module> parent_module;
    std::string name;
    std::tie(parent_module, name) = parseFullName(root_module, tensor_proto.name());

    auto param = buildParameter(tensor_proto);
    param_map[tensor_proto.name()] = &param;
    parent_module->register_parameter(name, param, tensor_proto.int64_data(0));
  }

  for (auto &node_proto : graph_proto.node()) {
    std::shared_ptr<script::Module> parent_module;
    std::string name;
    std::tie(parent_module, name) = parseFullName(root_module, node_proto.name());

    std::vector<at::Tensor*> member_inputs;
    for (auto &param_name : node_proto.input()) {
      member_inputs.push_back(param_map[param_name]);
    }

    auto graph = std::make_shared<Graph>();
    std::unordered_map<std::string, Value*> value_map;
    buildBlock(node_proto.attribute(0).g(), graph->block(), value_map);
    parent_module->create_method(name, graph, member_inputs);
  }

  return root_module;
}

}  // namespace

std::shared_ptr<Graph> ImportIRGraph(const std::string& serialized_graph,
                                     std::vector<at::Tensor>& initializers) {
  GraphDecoder decoder;
  return decoder.decode(serialized_graph, initializers);
}

std::shared_ptr<script::Module> ImportIRModule(
    const std::shared_ptr<script::Module> module,
    const std::string& serialized_module,
    const std::unordered_map<std::string, std::string>& storage_map) {
  ModuleDecoder decoder;
  return decoder.decode(module, serialized_module, storage_map);
}

}}
