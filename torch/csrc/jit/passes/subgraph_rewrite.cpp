#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/irparser.h>
#include <torch/csrc/jit/subgraph_matcher.h>

namespace torch {
namespace jit {

void SubgraphRewriter::RegisterDefaultPatterns() {
  // TODO: Add actual patterns (like Conv-Relu).
  RegisterRewritePattern(
      R"IR(
graph(%x, %w, %b):
  %c = aten::conv(%x, %w, %b)
  %r = aten::relu(%c)
  return (%r))IR",
      R"IR(
graph(%x, %w, %b):
  %r = aten::convrelu(%x, %w, %b)
  return (%r))IR");
}

RewritePatternDescr* SubgraphRewriter::RegisterRewritePattern(
    const std::string& pattern,
    const std::string& replacement) {
  patterns_.emplace_back(new RewritePatternDescr());
  RewritePatternDescr& d = *patterns_.back();
  d.pattern = pattern;
  script::parseIR(d.pattern, &d.pattern_graph, d.vmap);

  d.replacement = replacement;
  script::parseIR(d.replacement, &d.replacement_graph);
  return &d;
}

script::Module SubgraphRewriter::runOnModule(const script::Module& module) {
  nodes_to_delete_.clear();
  for (const auto& m : module.get_methods()) {
    auto g = m.function().graph();
    runOnGraph(g);
  }
  return module;
}

void SubgraphRewriter::runOnGraph(
    std::shared_ptr<Graph>& graph,
    const std::function<
        bool(const Match&, const std::unordered_map<std::string, Value*>&)>&
        filter,
    const std::function<void(const RewriteCallbackInfo&)>& rewrite_callback) {
  for (const std::unique_ptr<RewritePatternDescr>& pattern : patterns_) {
    rewriteSinglePatternOnGraph(graph, *pattern, filter);
  }
}

void SubgraphRewriter::rewriteSinglePatternOnGraph(
    std::shared_ptr<Graph>& graph,
    const RewritePatternDescr& pattern,
    const std::function<
        bool(const Match&, const std::unordered_map<std::string, Value*>&)>&
        filter,
    const std::function<void(const RewriteCallbackInfo&)>& rewrite_callback) {
  std::unordered_map<Value*, Value*> rewrite_map;
  std::vector<Value*> values_to_rewrite;

  const Graph& pattern_graph = pattern.pattern_graph;
  const auto& vmap = pattern.vmap;

  const Graph& replacement_graph = pattern.replacement_graph;

  const auto& matches = findPatternMatches(pattern_graph, *graph);
  for (const Match& match : matches) {
    if (!filter(match, vmap)) {
      continue;
    }
    // Matches might overlap with each other, in that case some of the nodes in
    // the current match might have already been used in another folded pattern.
    // We need to skip such matches.
    if (overlapsWithPreviousMatches(&match)) {
      continue;
    }

    // Figure out what values we need to use as inputs and outputs for the
    // replacement subgraph. These would be inputs and outputs of the subgraph
    // we matched.
    std::vector<Value*> inputs, outputs;
    for (const Value* v : pattern_graph.inputs()) {
      inputs.push_back(match.values_map.at(v));
    }
    for (const Value* v : pattern_graph.outputs()) {
      outputs.push_back(match.values_map.at(v));
    }

    // Insert a clone of replacement subgraph after the matched subgraph.
    // `inputs` vector holds values that we would use as incoming values to the
    // new subgraph, and we will get `new_outputs` vector containing values
    // produced by this new subgraph - we will then rewrite old outputs with the
    // new ones.
    WithInsertPoint insert_point(match.anchor);
    // TODO: const cast is only here because `insertGraph` unnecessary takes a
    // mutable reference
    std::vector<Value*> new_outputs =
        insertGraph(*graph, const_cast<Graph&>(replacement_graph), inputs);

    rewrite_callback(RewriteCallbackInfo{
        inputs, outputs, new_outputs, &pattern, &match});

    // Record all planned rewritings
    AT_ASSERT(outputs.size() == new_outputs.size());
    for (size_t idx = 0; idx < outputs.size(); idx++) {
      values_to_rewrite.push_back(outputs[idx]);
      rewrite_map[outputs[idx]] = new_outputs[idx];
    }
    // Record all planned deletions
    for (const Node* pattern_n : pattern_graph.nodes()) {
      if (match.nodes_map.count(pattern_n)) {
        Node* n = match.nodes_map.at(pattern_n);
        nodes_to_delete_.insert(n);
      }
    }
  }

  // Perform planned rewritings
  for (auto v : values_to_rewrite) {
    v->replaceAllUsesWith(rewrite_map.at(v));
  }

  // Perform planned deletions
  for (auto n : nodes_to_delete_) {
    n->removeAllInputs();
  }
  for (auto n : nodes_to_delete_) {
    n->destroy();
  }
}

bool SubgraphRewriter::overlapsWithPreviousMatches(const Match* match) {
  for (auto n : match->nodes_map) {
    if (nodes_to_delete_.count(n.second)) {
      return true;
    }
  }
  return false;
}

script::Module PatternBasedRewrite(const script::Module& module) {
  // TODO: Deep-copy the module
  SubgraphRewriter subgraph_rewriter;
  subgraph_rewriter.RegisterDefaultPatterns();
  return subgraph_rewriter.runOnModule(module);
}

} // namespace jit
} // namespace torch
