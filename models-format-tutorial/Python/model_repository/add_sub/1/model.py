import triton_python_backend_utils as pb_utils
import numpy as np

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            # Получаем входные данные
            input0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            input1 = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()

            # Выполняем операции сложения и вычитания
            add_result = input0 + input1
            sub_result = input0 - input1

            # Создаем выходные тензоры
            add_output = pb_utils.Tensor("ADD_OUTPUT", add_result)
            sub_output = pb_utils.Tensor("SUB_OUTPUT", sub_result)

            # Создаем ответ
            responses.append(pb_utils.InferenceResponse(output_tensors=[add_output, sub_output]))
        return responses

    def finalize(self):
        pass 