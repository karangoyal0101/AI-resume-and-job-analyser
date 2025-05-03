skill_keywords = [
    # Programming Languages
    "python", "java", "c++", "c", "javascript", "ruby", "go", "rust", "swift", "kotlin", 
    "typescript", "php", "shell scripting", "perl", "r", "matlab", "haskell", "objective-c", 
    "scala", "lua", "groovy", "elixir", "dart", "vhdl", "sql", "graphql", "json", "html", 
    "css", "xml", "yaml", "html5", "actionscript", "batch scripting", "xquery", "tcl",

    # Web Development Frameworks
    "django", "flask", "react", "angular", "vue.js", "node.js", "express.js", "ember.js", 
    "svelte", "next.js", "gatsby", "nuxt.js", "spring", "laravel", "rails", "zend", "backbone.js", 
    "bootstrap", "materialize", "tailwind css", "foundation", "apache", "nginx", "redis",

    # Front-End Development
    "html", "css", "javascript", "sass", "webpack", "babel", "typescript", "ajax", "jquery", 
    "responsive design", "cross-browser compatibility", "ui/ux design", "scss", "less", "react hooks", 
    "react native", "vuex", "redux", "material ui", "ionic", "swiftui", "flutter", "ember.js", 
    "angularjs", "css grid", "flexbox", "bootstrap", "tailwind", "javascript animations", "css animations",

    # Back-End Development
    "java", "python", "node.js", "c#", "go", "ruby", "php", "scala", "express.js", "spring", 
    "hibernate", "restful services", "microservices", "api development", "soap", "websockets", 
    "oauth2", "jwt", "grpc", "graphql", "mongodb", "postgresql", "mysql", "sqlite", "redis", 
    "couchdb", "firebase", "firebase cloud functions", "neo4j", "rabbitmq", "kafka", "message queuing",
    "apache kafka", "apache beam", "gcp", "aws lambda", "azure functions", "docker", "docker-compose", 
    "jenkins", "gitlab", "ci/cd", "jenkins pipelines", "cloudformation", "serverless architecture",

    # Data Engineering & Big Data
    "spark", "hadoop", "mapreduce", "kafka", "elasticsearch", "apache flink", "apache beam", 
    "storm", "kinesis", "bigquery", "etl", "etl pipelines", "data lakes", "data warehouses", 
    "big data analytics", "presto", "impala", "hive", "google cloud dataflow", "airflow", "dagster", 
    "data modeling", "aws s3", "hdfs", "parquet", "avro", "delta lake", "kubernetes", "containerization", 
    "cloud storage", "cloud computing", "docker", "pivotal cloud foundry", "databricks", "hadoop ecosystem", 
    "oracle", "cassandra", "couchbase", "sql server",

    # Machine Learning & AI
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras", "scikit-learn", 
    "xgboost", "lightgbm", "catboost", "reinforcement learning", "neural networks", "cnn", 
    "rnn", "lstm", "autoencoders", "gan", "svm", "decision trees", "random forest", "feature engineering", 
    "feature selection", "hyperparameter tuning", "cross-validation", "ensemble learning", "clustering", 
    "dimensionality reduction", "pca", "t-sne", "grid search", "bayesian optimization", "time series forecasting", 
    "computer vision", "opencv", "image processing", "nlp", "text classification", "sentiment analysis", 
    "chatbots", "speech recognition", "natural language generation", "word embeddings", "bert", "gpt", 
    "bert-based models", "transfer learning", "meta learning", "reinforcement learning", "model deployment", 
    "mlops", "ai ethics", "knowledge graphs", "semantic web", "data preprocessing", "data augmentation",

    # Cloud Technologies & DevOps
    "aws", "azure", "google cloud", "devops", "docker", "kubernetes", "terraform", "ansible", "chef", 
    "puppet", "ci/cd", "jenkins", "git", "gitlab", "github", "bitbucket", "cloudformation", "aws lambda", 
    "elastic beanstalk", "heroku", "monitoring", "prometheus", "grafana", "splunk", "new relic", 
    "datadog", "kibana", "logstash", "cloud security", "firewall", "cybersecurity", "vpn", "iso 27001", 
    "cloud native", "microservices architecture", "serverless architecture",

    # Databases & NoSQL
    "mysql", "postgresql", "mongodb", "redis", "cassandra", "dynamodb", "neo4j", "elasticsearch", 
    "sqlite", "firebase", "mariadb", "cockroachdb", "graphql", "sql server", "oracle", "bigtable", 
    "couchdb", "hbase", "amazon rds", "snowflake", "google cloud sql", "cloud firestore", "noSQL", "sqlite3", 
    "sqlalchemy", "dbeaver", "dbt", "indexeddb", "riak", "redis clusters",

    # Testing & Quality Assurance
    "unit testing", "integration testing", "test-driven development", "pytest", "mocha", "junit", 
    "selenium", "cypress", "karma", "chai", "enzyme", "jest", "supertest", "load testing", "performance testing", 
    "mocking", "ci testing", "continuous testing", "security testing", "functional testing", "regression testing", 
    "manual testing", "selenium grid", "mocking frameworks", "code coverage", "test automation",

    # Security & Cryptography
    "cybersecurity", "encryption", "ssl/tls", "public key infrastructure", "firewall", "vpn", "iso 27001", 
    "ethical hacking", "penetration testing", "network security", "application security", "oauth2", "jwt", 
    "rsa encryption", "aes encryption", "hashing algorithms", "security protocols", "xss", "sql injection", 
    "ddos protection", "sso", "incident response", "penetration testing", "bug bounty", "data protection", 
    "identity management", "siem", "compliance", "gdpr",

    # Blockchain & Cryptocurrency
    "blockchain", "ethereum", "smart contracts", "solidity", "cryptocurrency", "bitcoin", "ethereum", 
    "nft", "decentralized applications", "web3", "ipfs", "crypto trading", "cryptography", "defi", "dao", 
    "hyperledger", "ripple", "litecoin", "solana", "polkadot", "cosmos", "blockchain development", 
    "proof of stake", "proof of work", "consensus algorithms", "private blockchain",

    # Artificial Intelligence & Robotics
    "robotics", "robot operating system (ros)", "robot control", "path planning", "reinforcement learning", 
    "robotic process automation", "computer vision", "artificial intelligence", "autonomous systems", 
    "speech processing", "natural language processing", "machine perception", "deep reinforcement learning",

    # Other Specialized Tools & Topics
    "vfx", "3d modeling", "blender", "maya", "3ds max", "unreal engine", "unity", "augmented reality", 
    "virtual reality", "iot", "edge computing", "quantum computing", "algorithms", "data structures", 
    "parallel computing", "mapreduce", "cloud architecture", "event-driven architecture", "data visualization", 
    "tableau", "power bi", "matplotlib", "plotly", "seaborn", "ggplot2", "d3.js", "plotly.js", "kibana",
    "deep reinforcement learning", "human-computer interaction", "ar/vr development", "edge AI", "ML pipelines"
]


def extract_skills_from_jd(text, skill_list):
    text = text.lower()
    return [skill for skill in skill_list if skill in text]

import re

def extract_experience(text):
    match = re.search(r'(\d+)\s+year', text.lower())
    return f"{match.group(1)} years" if match else "Not specified"

def extract_degree(text):
    text = text.lower()
    if "bachelor" in text:
        return "Bachelor's"
    elif "master" in text:
        return "Master's"
    elif "phd" in text:
        return "PhD"
    else:
        return "Not specified"

import pandas as pd

df = pd.read_csv("job_title_des.csv")

structured = []
for i, row in df.iterrows():
    jd_text = str(row["Job Description"]).lower()
    
    skills = extract_skills_from_jd(jd_text, skill_keywords)
    exp = extract_experience(jd_text)
    degree = extract_degree(jd_text)
    job_title = row.get("Job Title", "Unknown")

    structured.append({
        "job_title": job_title,
        "required_skills": ", ".join(skills),
        "minimum_experience": exp,
        "required_degree": degree
    })

df_structured = pd.DataFrame(structured)
df_structured.to_csv("structured_job_descriptions_fast.csv", index=False)
print("✅ Fast processing complete!")
