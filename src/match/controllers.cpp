#include "controllers.hpp"

namespace pong {

bool isHumanSpec(const std::string& spec){ return spec == "human"; }

std::unique_ptr<Agent> makeAgent(const std::string& spec){
  if (spec == "p:easy")            return std::make_unique<PAgent>("P easy", 0.02, 0.55);
  if (spec == "p:classic")         return std::make_unique<PAgent>("P classic", 0.02, 0.75);  // ≈ old ui_pong AiVmax 0.9 cu/s
  if (spec == "p:hard")            return std::make_unique<PAgent>("P hard", 0.02, 1.0);
  if (spec == "interceptor")       return std::make_unique<InterceptorAgent>("Interceptor", 0.015, 1);
  if (spec == "interceptor:laggy") return std::make_unique<InterceptorAgent>("Interceptor laggy", 0.03, 12);
  // Held-out variants (DECISIONS.md #14): parameters no fitness game ever uses — the
  // generalization gauge once trained-on bots enter fitness. Eval/GUI only by convention.
  if (spec == "p:heldout")           return std::make_unique<PAgent>("P held-out", 0.05, 0.85);
  if (spec == "interceptor:heldout") return std::make_unique<InterceptorAgent>("Interceptor held-out", 0.03, 36);
  if (spec.rfind("mlp:", 0) == 0){
    auto a = std::make_unique<MlpAgent>("Model");
    if (!a->loadFile(spec.substr(4))) return nullptr;
    return a;
  }
  return nullptr;                                    // "human" or unknown
}

std::string agentLabel(const std::string& spec){
  if (spec == "human")             return "Human (keyboard)";
  if (spec == "p:easy")            return "P-controller (easy)";
  if (spec == "p:classic")         return "P-controller (classic)";
  if (spec == "p:hard")            return "P-controller (hard)";
  if (spec == "interceptor")       return "Interceptor";
  if (spec == "interceptor:laggy") return "Interceptor (laggy)";
  if (spec == "p:heldout")           return "P-controller (held-out)";
  if (spec == "interceptor:heldout") return "Interceptor (held-out)";
  if (spec.rfind("mlp:", 0) == 0)  return "Model: " + spec.substr(spec.find_last_of('/') + 1);
  return spec;
}

std::vector<std::string> builtinSpecs(){
  return {"p:easy", "p:classic", "p:hard", "interceptor", "interceptor:laggy"};
}

} // namespace pong
