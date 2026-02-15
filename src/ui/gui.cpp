#include "gui.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

GUI::GUI(Window& window) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  ImGui::StyleColorsDark();

  ImGui_ImplGlfw_InitForOpenGL(window.getWindow(), true);
  ImGui_ImplOpenGL3_Init("#version 330");
}

GUI::~GUI() {
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void GUI::prepare(ApplicationState& state, ApplicationRequests& requests) {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  createButtonRow(state);
  createCloseButton(requests);
}

void GUI::render() {
  ImGui::Render();
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void GUI::createButtonRow(ApplicationState& state) {
  ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowBgAlpha(0.0f);

  ImGui::Begin("Controls", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBackground);
  auto tool_button = [&state](char const* label, ToolState::Tool tool) {
    bool is_active = state.tool.activeTool == tool;
    if (is_active) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.6f, 0.2f, 1.0f));
    }
    if (ImGui::Button(label)) {
      state.tool.activeTool = is_active ? ToolState::Tool::NONE : tool;
    }
    if (is_active) {
      ImGui::PopStyleColor();
    }
  };

  tool_button("A", ToolState::Tool::A);
  ImGui::Spacing();
  tool_button("B", ToolState::Tool::B);
  ImGui::Spacing();
  tool_button("C", ToolState::Tool::C);

  ImGui::End();
}

void GUI::createCloseButton(ApplicationRequests& requests) {
  ImGuiIO& io = ImGui::GetIO();
  float window_width = io.DisplaySize.x;

  ImGui::SetNextWindowPos(ImVec2(window_width - 50, 10), ImGuiCond_Always);
  ImGui::SetNextWindowBgAlpha(0.0f);

  ImGui::Begin("CloseButton", nullptr,
               ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoBackground);

  if (ImGui::Button("X")) {
    requests.shutdown = true;
  }

  ImGui::End();
}
