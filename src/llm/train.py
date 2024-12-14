import torch
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

sys_prompt = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n
                ### Инструкция:
                Определи, используя знания о человеческой психологии, из какой одной категории товаров, существующих в нашем интернет магазине, пользователь купит следующий товар
                ### Вход:
                {input}
                ### Ответ:'''

def init_llama3():
    MODEL_NAME = "t-tech/T-lite-it-1.0"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )    
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer

    
def train(model, tokenizer , dataset, collator = None, epoches_count = 1):
    lora_config = LoraConfig(
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=0.05
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.train()
    
    train_args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        #num_train_epochs = epoches_count, # Set this for 1 full training run.
        max_steps = 750,
        learning_rate = 3e-4,
        #fp16 = not is_bfloat16_supported(),
        #bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset["train"],
        eval_dataset = None,
        dataset_text_field = "text",
        max_seq_length = 2000,
        dataset_num_proc = 2,
        data_collator = collator,
        packing = False, # Can make training 5x faster for short sequences.
        args = train_args)

    trainer_stats = trainer.train()
    model.save_pretrained("model/")
    return trainer_stats

if __name__ == "__main__":
    dataset = GPTDataset()
    model, tokenizer = init_llama3()
    train(model,tokenizer, dataset)
    
    