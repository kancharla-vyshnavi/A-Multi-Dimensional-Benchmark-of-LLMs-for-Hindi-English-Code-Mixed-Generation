import pandas as pd
import re
import os

INPUT = "outputs/controlled_generation/benchmark_generations_v3.csv"
OUT_DIR = "outputs/hinglish_metrics_v3"
OUTPUT = f"{OUT_DIR}/code_mixing_index_v3.csv"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT)

# Common English function/content words.
# This is a transparent heuristic, not a full language-identification system.
ENGLISH_WORDS = {
    "a","an","the","and","or","but","if","then","is","are","was","were",
    "am","be","been","being","to","of","in","on","at","for","from","with",
    "without","about","into","over","under","this","that","these","those",
    "it","its","i","you","he","she","we","they","my","your","our","their",
    "can","could","will","would","should","may","might","do","does","did",
    "have","has","had","not","no","yes","very","more","most","some","any",
    "all","one","two","three","what","when","where","why","how","who",
    "good","bad","new","old","big","small","make","made","use","used",
    "need","want","like","know","think","go","come","get","give","take",
    "please","thanks","thank","hello","sorry","because","also","only",
    "just","really","important","problem","answer","example","time",
    "people","work","school","home","day","today","tomorrow","yesterday"
}

# Common Romanized Hindi words.
HINDI_WORDS = {
    "hai","hain","tha","the","thi","ho","hoga","hogi","honge",
    "ka","ki","ke","ko","se","me","mein","par","pe","ye","yah","yeh",
    "woh","vo","aur","ya","lekin","kyunki","kyun","kya","kaise","kab",
    "kahan","kaun","mujhe","mujh","mera","meri","mere","tum","tumhe",
    "aap","aapko","hum","hume","hamara","hamari","unka","unki","unke",
    "is","us","in","un","ek","do","bhi","hi","toh","to","nahi","nahin",
    "mat","haan","ji","ab","phir","fir","bahut","zyada","kam",
    "accha","achha","acha","acchi","achhi","achhe","bura","buri",
    "kar","karo","karna","karte","karti","kiya","kiye","gaya","gayi",
    "jao","jana","jaana","aao","aana","ao","bolo","bol","dekho","dekh",
    "sun","suno","samajh","samajhna","chahiye","sakta","sakti","sakte",
    "raha","rahi","rahe","wala","wali","wale","keval","sirf",
    "kyon","kyonki","jab","jabki","agar","magar","abhi","yahan","wahan"
}

def classify_word(word):
    word = word.lower()

    if word in ENGLISH_WORDS:
        return "english"

    if word in HINDI_WORDS:
        return "hindi"

    return "other"


def calculate(text):
    text = str(text).lower()

    # Roman/Latin word tokens only.
    words = re.findall(r"[a-z]+", text)

    labels = [classify_word(w) for w in words]

    english = labels.count("english")
    hindi = labels.count("hindi")
    classified = english + hindi

    # CMI-style measure:
    # proportion of classified words belonging to the minority language.
    if classified > 0:
        minority = min(english, hindi)
        cmi = (minority / classified) * 100
    else:
        cmi = 0

    return pd.Series({
        "total_roman_words": len(words),
        "english_lexicon_words": english,
        "hindi_lexicon_words": hindi,
        "classified_words": classified,
        "english_share_classified": english / classified if classified else 0,
        "hindi_share_classified": hindi / classified if classified else 0,
        "code_mixing_index_percent": cmi,
        "both_languages_present": int(english > 0 and hindi > 0)
    })


metrics = df["response"].apply(calculate)

result = pd.concat(
    [df[["row_index", "model"]], metrics],
    axis=1
)

result.to_csv(OUTPUT, index=False)

summary = (
    result.groupby("model")
    .agg(
        responses=("row_index", "count"),
        avg_cmi_percent=("code_mixing_index_percent", "mean"),
        avg_english_share=("english_share_classified", "mean"),
        avg_hindi_share=("hindi_share_classified", "mean"),
        both_languages_rate=("both_languages_present", "mean"),
        avg_classified_words=("classified_words", "mean")
    )
    .reset_index()
)

SUMMARY = f"{OUT_DIR}/code_mixing_summary_v3.csv"
summary.to_csv(SUMMARY, index=False)

print("\n=== CODE-MIXING INDEX ===")
print(summary.to_string(index=False))

print("\nCreated:")
print(OUTPUT)
print(SUMMARY)

print("\nNOTE:")
print("CMI here is a transparent lexicon-based heuristic.")
print("It should be reported as an approximate lexical mixing measure,")
print("not as a gold-standard language identification score.")
