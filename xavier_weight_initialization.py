# Sometimes we need to change pre-trained model architecture to suit our use cases
# For example the pre-trained architecture trained with 100 Classes to classify like in Image Net Database
# But our use case contains only 5 cases.
# In such cases we need to modify the final classification layer either throug the model config class or through directly accessing the model.cls_head.
# In these cases after changing the number of nodes, the correonsing weights are randomly initlialized, or not initialized at all.
# Hence, loss becomes extremely high (NaN), to fix that we need to do weight_initialization as below.
# This is a sample code, modify it accordng to your model architecture and dataset.

from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import ViltConfig

config = ViltConfig.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
model = ViltForQuestionAnswering.from_pretrained("../model_chkpts/best_vilt_chkpt_with_1_fact_64_esnli", id2label = reverse_mapping, label2id = mapping, ignore_mismatched_sizes=True).to(device)

# Define the Xavier initialization function

def initialize_weights(m):

    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

# Apply the weight initialization specifically to the last layer

model.classifier[-1].apply(initialize_weights)
