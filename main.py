import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import random

# Step 1: Generate synthetic dataset with 1,000 examples
def generate_synthetic_data(n_samples=1000):
    cta_templates = [
        ("Click {action} to {benefit}!", 1, 90),
        ("Join {group} {time}!", 1, 85),
        ("Buy {product} and get {incentive}!", 1, 95),
        ("Sign up for {offer} {time}.", 1, 80),
        ("Hurry, {offer} ends {time}!", 1, 92),
        ("Get {benefit} by {action} now!", 1, 88),
        ("Subscribe to {service} for {benefit}!", 1, 85),
        ("Don’t miss out, {action} {time}!", 1, 90),
        ("Register for {event} today!", 1, 88),
        ("Start your {offer} now!", 1, 87),
        ("You might want to {action} this.", 1, 60),
        ("Consider {action} for {benefit}.", 1, 55),
    ]
    non_cta_templates = [
        ("Check out {product}.", 0, 10),
        ("{product} is {adjective}.", 0, 15),
        ("Learn more about {topic}.", 0, 20),
        ("Our {product} offers {feature}.", 0, 25),
        ("Explore {topic} with us.", 0, 30),
        ("{product} has {feature}.", 0, 20),
        ("We provide {service}.", 0, 15),
        ("Discover {topic} today.", 0, 25),
    ]

    actions = ["here", "now", "today", "this link", "our site"]
    benefits = ["save 20%", "free shipping", "exclusive access", "a discount", "early access"]
    groups = ["our newsletter", "the community", "our club", "the team"]
    times = ["today", "now", "this week", "soon"]
    products = ["our product", "this deal", "our new line", "the latest gadget"]
    incentives = ["free shipping", "a bonus gift", "50% off", "a free trial"]
    offers = ["a free trial", "our plan", "this deal", "our service"]
    events = ["our webinar", "the event", "our workshop", "the conference"]
    services = ["updates", "our newsletter", "exclusive content", "our platform"]
    adjectives = ["awesome", "great", "amazing", "reliable", "top-quality"]
    topics = ["our services", "the new collection", "our story", "this topic"]
    features = ["great quality", "new features", "fast delivery", "unique design"]

    messages, labels, scores = [], [], []
    for i in range(n_samples):
        if random.random() < 0.6:  # 60% CTA messages
            template, label, base_score = random.choice(cta_templates)
            message = template.format(
                action=random.choice(actions),
                benefit=random.choice(benefits),
                group=random.choice(groups),
                time=random.choice(times),
                product=random.choice(products),
                incentive=random.choice(incentives),
                offer=random.choice(offers),
                event=random.choice(events),
                service=random.choice(services)
            )
            score = base_score + random.randint(-5, 5)  # Add variation
        else:  # 40% non-CTA messages
            template, label, base_score = random.choice(non_cta_templates)
            message = template.format(
                product=random.choice(products),
                adjective=random.choice(adjectives),
                topic=random.choice(topics),
                feature=random.choice(features),
                service=random.choice(services)
            )
            score = base_score + random.randint(-5, 5)  # Add variation
        
        # Ensure score stays in 0-100
        score = max(0, min(100, score))
        messages.append(message)
        labels.append(label)
        scores.append(score)

    return pd.DataFrame({"message": messages, "label": labels, "cta_score": scores})

# Generate dataset
df = generate_synthetic_data(1000)

# Your 18 test messages (transcribed from screenshot)
test_messages = [
    "Find the perfect way to connect with others",
    "Never worry about where to meet again",
    "Say goodbye to back-and-forth scheduling",
    "Share your schedule without overstepping",
    "Book more of your most valuable meetings",
    "Use event types to simplify your scheduling",
    "Keep your day organized without the hassle",
    "How to run better meetings",
    "Optimize your meetings and free up your day",
    "Simplify event scheduling for your whole team",
    "Choose the right format for every meeting",
    "Master the art of professional availability",
    "Your guide to the perfect scheduling setup",
    "Keep your team aligned with less effort",
    "Automate scheduling and reclaim your focus",
    "Keep meetings on track with fewer no-shows",
    "Take control of your day—on your terms",
    "Free up time by automating your day"
]

# Step 2: Custom Dataset class for BERT
class CTADataset(Dataset):
    def __init__(self, messages, labels, tokenizer, max_len=128):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx):
        message = str(self.messages[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Step 3: Prepare data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_messages, val_messages, train_labels, val_labels = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

train_dataset = CTADataset(train_messages.tolist(), train_labels.tolist(), tokenizer)
val_dataset = CTADataset(val_messages.tolist(), val_labels.tolist(), tokenizer)

# Step 4: Initialize BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Step 5: Training arguments
training_args = TrainingArguments(
    output_dir='./cta_bert_results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./cta_bert_logs',
    logging_steps=10,
    eval_steps=500,  # Evaluate every 500 steps
    save_strategy='epoch',  # Save at the end of each epoch
    # Removed load_best_model_at_end to avoid conflict
)

# Step 6: Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)
trainer.train()

# Step 7: Inference on 18 test messages
model.eval()
test_scores = []

for message in test_messages:
    inputs = tokenizer(message, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    cta_prob = probs[0][1].item()  # Probability of CTA (class 1)
    cta_score = round(cta_prob * 100, 2)  # Scale to 0-100
    test_scores.append((message, cta_score))

# Step 8: Output results
results = pd.DataFrame(test_scores, columns=['Message', 'CTA_Score'])
print("\nCTA Scores for Test Messages:")
print(results)

# Save results to CSV
results.to_csv('cta_test_results.csv', index=False)
