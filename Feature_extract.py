import pandas as pd
import re
import tldextract

# Load dataset
df = pd.read_csv("labeled_dataset.csv")

# List of suspicious keywords
suspicious_keywords = ["login", "secure", "bank", "account", "verify", "payment"]

# Function to extract features
def extract_url_features(url):
    domain_info = tldextract.extract(url)
    return {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_slashes": url.count("/"),
        "num_hyphens": url.count("-"),
        "num_digits": sum(c.isdigit() for c in url),
        "contains_at": "@" in url,
        "https_in_domain": "https" in domain_info.domain,
        "subdomain_length": len(domain_info.subdomain),
        "tld": domain_info.suffix,
        "contains_suspicious_word": any(word in url.lower() for word in suspicious_keywords)
    }

# Apply function to extract features
df_features = df["URL"].apply(lambda x: extract_url_features(x)).apply(pd.Series)

# Save extracted features
df_final = pd.concat([df, df_features], axis=1)
df_final.to_csv("processed_dataset.csv", index=False)

print(f"✅ Processed dataset with URL-based features saved as 'processed_dataset.csv'.")
