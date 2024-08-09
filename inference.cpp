#include <iostream>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner_cpu.h>

int main(int argc, char *argv[]) {
    
    char *model_path = NULL;  // e.g. mode.so
    model_path = argv[1];

    c10::InferenceMode mode;
    torch::inductor::AOTIModelContainerRunnerCpu *runner;
    runner = new torch::inductor::AOTIModelContainerRunnerCpu(model_path, 1);
    std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCPU)};
    std::vector<torch::Tensor> outputs = runner->run(inputs);
    std::cout << "Result from the first inference:"<< std::endl;
    std::cout << outputs[0] << std::endl;

    // The second inference uses a different batch size and it works because we
    // specified that dimension as dynamic when compiling model.so.
    std::cout << "Result from the second inference:"<< std::endl;
    std::vector<torch::Tensor> inputs2 = {torch::randn({2, 10}, at::kCPU)};
    std::cout << runner->run(inputs2)[0] << std::endl;

    return 0;
}
