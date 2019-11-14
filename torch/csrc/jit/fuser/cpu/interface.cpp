#include <torch/csrc/jit/fuser/cpu/interface.h>
#include <torch/csrc/jit/fuser/common/utils.h>

#include <asmjit/asmjit.h>
#include <iostream>

#define FUSER_DEBUG 1

namespace torch {
namespace jit {
namespace fuser {
namespace cpu {

using namespace torch::jit::fuser;

// Signature of the generated function.
typedef int (*Func)(void);
typedef void (*addFunc)(unsigned, float*, float*, float*);

using namespace asmjit;
using namespace asmjit::x86;

// Returns true if the node is added to the fusion group, false o.w.
bool mergeNodeWithFusionGroup(const Node* const node, Node* fusion_group) {
  #if FUSER_DEBUG
    std::cout << "cpuMergeNodeWithFusionGroup" << std::endl;
  #endif // FUSER_DEBUG

  // Creates a new fusion group
  if (fusion_group == nullptr) {
    // Validates inputs are fusible
    const auto inputs = node->inputs();
    const auto output = node->output();

    const auto lhs = inputs[0]->type()->expect<TensorType>();
    const auto rhs = inputs[1]->type()->expect<TensorType>();
    const auto c = inputs[2]; // TODO: validate c = 1

    const auto lhs_rank = getRank(lhs);
    const auto rhs_rank = getRank(rhs);

    if (lhs_rank != rhs_rank) {
      std::cout << "Rank mismatch!" << std::endl;
    }

    const auto lhs_dims = getNumNonCollapsibleDims(lhs);
    const auto rhs_dims = getNumNonCollapsibleDims(rhs);

    if (lhs_dims != rhs_dims) {
      std::cout << "Dims mismatch!" << std::endl;
    }

    const auto lhs_numel = getNumel(lhs);
    const auto rhs_numel = getNumel(rhs);

    if (lhs_numel != rhs_numel) {
      std::cout << "numel mismatch!" << std::endl;
    }

    // Creates fusion_group

    // Creates runtime and assembler
    JitRuntime rt;
    CodeHolder code;
    code.init(rt.codeInfo());
    Assembler a(&code);

    // Creates fusion (element-by-element contiguous add)
    Label LoopInc = a.newLabel();
    Label LoopBody = a.newLabel();
    Label Exit = a.newLabel();

    // Short-circuits on size == 0
    a.test(edi, edi);
    a.je(Exit);

    // Stores # of loop iterations, sets loop counter to zero
    a.lea(r8d, dword_ptr(rdi, - 1)); // r8 = size - 1
    a.xor_(eax, eax); // clears eax
    a.jmp(LoopBody); // do () { } while () loop form

    // Loop incrementer
    a.bind(LoopInc);
    a.mov(rax, rdi); // offset = offset + 1

    // Loop body
    a.bind(LoopBody);
    a.vmovss(xmm0, dword_ptr(rdx, rax, 2)); // xmm0 = lhs[offset]
    a.vaddss(xmm0, xmm0, dword_ptr(rcx, rax, 2)); // xmm0 = xmm0 + rhs[offset]
    a.lea(rdi, dword_ptr(rax, 1)); // size = offset + 1
    a.vmovss(dword_ptr(rsi, rax, 2), xmm0); // out[offset] = xmm0

    // Checks if loop is finished
    a.cmp(rax, r8); // if offset == size - 1, terminate
    a.jne(LoopInc);

    // Exit
    a.bind(Exit);
    a.ret();

    // Jits the code and stores in function
    addFunc fn;
    Error err = rt.add(&fn, &code);
    if (err) {
      std::cout << "Error while jitting!" << std::endl;
    }

    return true;
  }

  return false;
}

}}}
} // namespace torch
