#pragma once
#include "agents.hpp"
#include <memory>
#include <string>
#include <vector>

namespace pong {

// Controller specs — the serializable controller identity shared by the GUI, the match
// runner, and the future league (Elo keys, hall-of-fame refs, match metadata). Grammar:
//   "human"              keyboard-driven side (GUI only; makeAgent returns nullptr)
//   "p:easy|classic|hard"        PAgent difficulty presets (duty 0.55 / 0.75 / 1.0)
//   "interceptor" | "interceptor:laggy"   trajectory predictor (instant / 200 ms replan)
//   "mlp:<path>"         MlpAgent loaded from an aipong-mlp v1/v2 weight file
bool isHumanSpec(const std::string& spec);

// Fresh agent instance per call (agents are stateful — never share across sides).
// nullptr for "human", unknown specs, or a failed model load.
std::unique_ptr<Agent> makeAgent(const std::string& spec);

// Stable display label; embedded verbatim in recorded match metadata, so changing a
// label is a data-compatibility decision, not cosmetics.
std::string agentLabel(const std::string& spec);

// The scripted roster in display order (excludes "human" and mlp models).
std::vector<std::string> builtinSpecs();

} // namespace pong
