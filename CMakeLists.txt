cmake_minimum_required(VERSION 3.22)
project(ai_pong CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # static libs feed the pybind11 module → PIC everywhere
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)
option(AIPONG_MARCH_NATIVE "Tune codegen for the build machine's CPU (non-portable binaries)" ON)
if(AIPONG_MARCH_NATIVE)
  add_compile_options(-march=native)
endif()

include(FetchContent)
# Pinned deps. Verify a tag exists with: git ls-remote --tags <repo>
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest GIT_TAG v2.4.11)
FetchContent_MakeAvailable(doctest)
FetchContent_Declare(imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui GIT_TAG v1.90.4)
FetchContent_MakeAvailable(imgui)   # source-only (no CMake); compile the units we need
FetchContent_Declare(pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11 GIT_TAG v2.12.0)
FetchContent_MakeAvailable(pybind11)

add_library(pong_core STATIC src/core/pong_core.cpp)
target_include_directories(pong_core PUBLIC src/core)

# -ffast-math scoped to the MLP hot path only (physics keeps strict FP): permits the
# dot-product reduction to vectorize and libmvec's SIMD tanh. Within-binary determinism
# is unaffected; see DECISIONS.md #11 for the measurement story.
set_source_files_properties(src/agents/agents.cpp PROPERTIES COMPILE_OPTIONS "-ffast-math")
add_library(pong_agents STATIC src/agents/agents.cpp)
target_include_directories(pong_agents PUBLIC src/agents)
target_link_libraries(pong_agents PUBLIC pong_core)

add_library(pong_record STATIC src/record/recorder.cpp src/record/rally_index.cpp)
target_include_directories(pong_record PUBLIC src/record)
target_link_libraries(pong_record PUBLIC pong_core)

# match/: controller spec registry + the one true step loop (MatchRunner, header-only) —
# shared by the GUI and the (future) headless league runner.
add_library(pong_match STATIC src/match/controllers.cpp)
target_include_directories(pong_match PUBLIC src/match)
target_link_libraries(pong_match PUBLIC pong_agents pong_record)

# train/: GA self-play trainer (evolution track). batch_forward gets the same scoped
# fast-math as agents.cpp — the lane loop is the SIMD dimension (Decision 13).
find_package(Threads REQUIRED)
set_source_files_properties(src/train/batch_forward.cpp PROPERTIES COMPILE_OPTIONS "-ffast-math")
add_library(pong_train STATIC src/train/evolve.cpp src/train/batch_forward.cpp src/train/coevolve.cpp)
target_include_directories(pong_train PUBLIC src/train)
target_link_libraries(pong_train PUBLIC pong_match Threads::Threads)

# league/: Elo ladder over checkpoints + anchors (the deployment-champion selector).
add_library(pong_league STATIC src/league/ladder.cpp)
target_include_directories(pong_league PUBLIC src/league)
target_link_libraries(pong_league PUBLIC pong_match Threads::Threads)

enable_testing()
function(aipong_test name)
  add_executable(${name} src/tests/${name}.cpp)
  target_link_libraries(${name} PRIVATE ${ARGN} doctest::doctest)
  add_test(NAME ${name} COMMAND ${name})
endfunction()
aipong_test(test_core   pong_core)
aipong_test(test_agents pong_agents)
aipong_test(test_record pong_agents pong_record)
aipong_test(test_match  pong_match)
aipong_test(test_train  pong_train)
aipong_test(test_league pong_league)
aipong_test(test_trainview pong_train)   # log-tail data layer for the live dashboard

# --- imgui frontends (immediate-mode native; GLFW + OpenGL3) ---
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
add_library(imgui_lib STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp        ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui_lib PUBLIC glfw OpenGL::GL ${CMAKE_DL_LIBS})

add_executable(pong_imgui src/renderers/imgui/main.cpp)     # the game arena
target_link_libraries(pong_imgui PRIVATE pong_match pong_league imgui_lib)

add_executable(pong_trainview src/renderers/imgui/train_view.cpp)   # live training dashboard
target_include_directories(pong_trainview PRIVATE src/train)
target_link_libraries(pong_trainview PRIVATE imgui_lib)

add_executable(pong_coview src/renderers/imgui/coview.cpp)
target_include_directories(pong_coview PRIVATE src/train)
target_link_libraries(pong_coview PRIVATE imgui_lib)

# --- tools ---
add_executable(gen_random_model src/tools/gen_random_model.cpp)
add_executable(rallydb src/tools/rallydb.cpp)
target_link_libraries(rallydb PRIVATE pong_record)
add_executable(evolve src/tools/evolve.cpp)
target_link_libraries(evolve PRIVATE pong_train)
add_executable(coevolve src/tools/coevolve.cpp)
target_link_libraries(coevolve PRIVATE pong_train)
add_executable(bench src/tools/bench.cpp)
target_link_libraries(bench PRIVATE pong_train)
add_executable(ladder src/tools/ladder.cpp)
target_link_libraries(ladder PRIVATE pong_league)
add_executable(aiming src/tools/aiming.cpp)
target_link_libraries(aiming PRIVATE pong_match)
add_executable(memuse src/tools/memuse.cpp)
target_link_libraries(memuse PRIVATE pong_match)
add_executable(reeval src/tools/reeval.cpp)
target_link_libraries(reeval PRIVATE pong_train)   # rng.hpp (train include dir) + match libs

# --- python bindings (PPO track): build/aipong.cpython-*.so, imported by src/train/ppo.py ---
pybind11_add_module(aipong src/bindings/aipong_module.cpp)
target_link_libraries(aipong PRIVATE pong_match)
