"""
This module performs product review aggregation and summary generation
using a pre-trained model from Hugging Face. The code includes functions for
processing product data, generating summaries, and deploying a Gradio interface.
"""

# %% Imports
import re
import random
import pandas as pd  # pylint: disable=E0401
import torch  # pylint: disable=E0401
from huggingface_hub import login  # pylint: disable=E0401
from transformers import LlamaTokenizer, AutoModelForCausalLM  # pylint: disable=E0401
import gradio as gr  # pylint: disable=E0401

# %% Constants
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
CSV_FILE = "filtered_data_predictions_clusters.csv"

# %% Hugging Face Login
login(token=HUGGINGFACE_TOKEN)

# %% Model and Tokenizer Setup
tokenizer = LlamaTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
torch.cuda.empty_cache()

# %% Load and Process Data
df = pd.read_csv(CSV_FILE)
df = df.drop(columns=["brand", "categories", "primaryCategories", "reviews.date"])

# Define cluster names for user-friendly selection
CLUSTER_NAMES = {
    0: "Smart Speakers",
    1: "Pet Supplies & AAA Batteries",
    2: "Fire Tablets & Streaming Devices with Alexa",
    3: "Fire Tablets - Standard Editions",
    4: "Fire Tablets - Kids Editions",
    5: "Fire Tablets with Alexa",
    6: "Alexa Devices & Accessories",
    7: "AA Batteries",
    8: "Kindle E-Readers & Accessories",
    9: "Echo Devices - Various Generations"
}

# Add cluster name column
df['category_name'] = df['cluster'].map(CLUSTER_NAMES)

# Review Summary Template
REVIEW_PROMPT_TEMPLATE = (
    "Provide a two-sentence summary of the main strengths and weaknesses "
    "for this product based on user reviews: {}"
)

# %% Functions
def top_product_summaries(results):
    """
    Prints summarized results for top products.
    """
    print("\n Here are the top products of the category:")
    print("=" * 60)
    for entry in enumerate(results, 1):
        print(f"\nProduct Name: {entry['product_name']}")
        print(f"Title Summary:\n  {entry['title_summary']}")
        print(f"Review Summary:\n  {entry['review_summary']}")
        print("=" * 60)


def generate_summary(prompt, max_new_tokens=20, temperature=0.5,
                     repetition_penalty=1.8, is_title=False):
    """
    Generates a summary or title based on the given prompt.
    """
    if is_title:
        max_new_tokens = 50
        temperature = 5

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        num_beams=3,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def post_process(text):
    """
    Post-processes generated text to remove prompt artifacts and enhance variety.
    """
    prompt_patterns = [
        r"^Create a one-sentence title summarizing.*:?\s*",
        r"^Provide a two-sentence summary.*:?\s*",
        r"^Focus on unique qualities only.*:?\s*",
        r"^Generate a 5-word title that summarizes the key.*:?\s*",
        r"^.*summarizing.*:\s*",
        r"^.*summary.*:\s*",
        r"^.*qualities only.*:\s*",
        r"^.*summarizes the key.*:\s*",
        r"^Strengths:\s*",
        r"^Weaknesses:\s*",
        r"\s*Strengths:\s*",
        r"\s*Example:\s*",
        r"\s*Review:\s*",
        r"^Examples?:\s*",
        r"^- Review\s*:\s*",
        r"^Main Weaknesses?:\s*",
        r"^\d+\.?",
        r"^-",
        r"^\s*"
    ]
    for pattern in prompt_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()

    last_period_index = text.rfind(".")
    if last_period_index != -1:
        text = text[:last_period_index + 1]

    sentences = re.split(r'(?<=\w[.!?])\s+', text)
    sentences = [s for s in sentences if s and len(s.split()) > 3]

    synonyms = {
        "great": ["excellent", "outstanding", "impressive"],
        "good": ["satisfactory", "decent", "suitable"],
        "easy": ["straightforward", "simple", "user-friendly"],
        "awesome": ["fantastic", "remarkable", "incredible"],
        "price": ["cost", "value", "affordability"],
    }

    for i, sentence in enumerate(sentences):
        for word, syn_list in synonyms.items():
            if word in sentence:
                synonym = random.choice(syn_list)
                sentence = re.sub(rf'\b{word}\b', synonym, sentence, flags=re.IGNORECASE)
        sentences[i] = sentence

    final_sentences = []
    for sentence in sentences:
        if "kids" in sentence.lower() or "children" in sentence.lower():
            sentence += " Ideal for young users due to its durable and easy-to-use design."
        elif "waterproof" in sentence.lower() or "beach" in sentence.lower():
            sentence += " Perfect for reading outdoors or by the pool."
        final_sentences.append(sentence)

    return " ".join(final_sentences)


def post_process_title(text):
    """
    Post-processes generated title text to clean artifacts and ensure completeness.
    """
    last_title_index = text.rfind("Title:")
    if last_title_index != -1:
        text = text[last_title_index + len("Title:"):].strip()

    lines = text.splitlines()
    prompt_patterns = [
        r"^Examples?:\s*$",
        r"^- Review\s*:\s*$",
        r"^Main Weaknesses?:\s*$",
        r"^Strengths?:\s*$",
        r"^\d+\.\s*$",
        r"^-\s*$",
        r"^\s*$"
    ]

    cleaned_lines = []
    for line in lines:
        if not any(re.match(pattern, line.strip(), re.IGNORECASE) for pattern in prompt_patterns):
            cleaned_lines.append(line.strip())

    processed_text = " ".join(cleaned_lines).strip()
    return processed_text.strip('\'"').rsplit('.', 1)[0] + '.'


def generate_summary_again(review_text, max_new_tokens=80):
    """
    Generates a more diverse summary with a higher temperature setting.
    """
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        repetition_penalty=1.2,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()


def display_top_products(selected_category):
    """
    Displays the top products in a selected category by generating summaries for them.
    """
    category_df = df[df['category_name'] == selected_category]

    filtered_df = category_df[
        (category_df['reviews.rating'].isin([1, 2, 3, 4, 5])) &
        (category_df['predicted_sentiment'] == category_df['sentiment']) &
        (
            ((category_df['reviews.rating'].isin([1, 2])) &
             (category_df['sentiment'] == 'negative')) |
            ((category_df['reviews.rating'] == 3) & (category_df['sentiment'] == 'neutral')) |
            ((category_df['reviews.rating'].isin([4, 5])) &
             (category_df['sentiment'] == 'positive'))
        )
    ]


    top_products = filtered_df.sort_values(
        by=['appearances', 'reviews.rating'],
        ascending=[False, False]
    ).drop_duplicates(subset='name').head(3)

    results = []
    for _, row in top_products.iterrows():
        product_name = row["name"]
        combined_text = row["reviews.title"] + " " + row["reviews.text"]
        review_prompt = REVIEW_PROMPT_TEMPLATE.format(combined_text[:150])
        review_summary = generate_summary(review_prompt, max_new_tokens=60)
        review_summary = post_process(review_summary)

        if not review_summary:
            review_prompt_2 = REVIEW_PROMPT_TEMPLATE.format(combined_text[150:300])
            review_summary = generate_summary(review_prompt_2, max_new_tokens=80)
            review_summary = post_process(review_summary)

        EXAMPLE_TITLES = """
        Examples:
        - Review: "This tablet has amazing battery life and is very user-friendly."
        Title: "Long Battery Life and Easy to Use"
        - Review: "The Kindle Fire is perfect for kids, durable, and affordable."
        Title: "Kid-Friendly, Durable, and Affordable"
        - Review: "Echo Show's sound quality is fantastic, but privacy concerns remain."
        Title: "Great Sound, But Privacy Concerns"
        """
        TITLE_PROMPT = (
            f"{EXAMPLE_TITLES}\nCreate a concise and catchy title that captures the main idea "
            f"of this review: {review_summary} Title:"
        )
        title_summary_raw = generate_summary(TITLE_PROMPT, is_title=True)
        title_summary = post_process_title(title_summary_raw)

        results.append(
            f"**{product_name}**\n\n**Title Summary:** {title_summary}\n\n"
            f"**Review Summary:** {review_summary}\n\n"
        )

    return "\n".join(results)


# %% Gradio Interface
interface = gr.Interface(
    fn=display_top_products,
    inputs=gr.Dropdown(choices=list(CLUSTER_NAMES.values()), label="Choose a category:"),
    outputs=gr.Markdown(label="Top Products Summary"),
    title="SenCluSum AI",
    description="Find out the top products and what users think of them! Make your choice"
)

# %% Launch Gradio App
interface.launch(share=True)
