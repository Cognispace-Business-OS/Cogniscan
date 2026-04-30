skills = [
    # Programming Languages
    "Python", "C++", "Java", "JavaScript", "TypeScript", "Go", "Rust", "Kotlin", "Swift", "Dart",
    "C", "C#", "PHP", "Ruby", "Scala", "MATLAB", "R", "Julia", "Bash", "Shell Scripting",
    "Zig", "Solidity", "Mojo", "Lua",

    # Core CS Fundamentals
    "Data Structures", "Algorithms", "Object-Oriented Programming", "Functional Programming",
    "Operating Systems", "Computer Networks", "Database Management Systems", "System Design",
    "Distributed Systems", "Compiler Design", "Concurrency Control", "Memory Management",
    "Computer Architecture", "Theory of Computation",

    # Backend Development
    "Django", "FastAPI", "Flask", "Spring Boot", "Node.js", "Express.js", "NestJS", "Go-kit",
    "REST APIs", "GraphQL", "gRPC", "Authentication Systems", "OAuth 2.0", "OIDC", "JWT",
    "Microservices Architecture", "Event-Driven Architecture", "Saga Pattern", "CQRS",
    "API Gateway (Kong, Tyk)", "Rate Limiting", "Caching Strategies", "Session Management",
    "WebSockets", "WebRTC", "Server-Sent Events (SSE)",

    # Frontend Development
    "HTML5", "CSS3", "JavaScript DOM", "React.js", "Next.js", "Vue.js", "Angular",
    "Tailwind CSS", "Bootstrap", "Sass/SCSS", "Web Performance Optimization", "Responsive Design",
    "Web Accessibility (A11y)", "State Management (Redux, Zustand, Recoil)", "Micro-Frontends",
    "WebAssembly (WASM)", "Progressive Web Apps (PWA)", "Three.js",

    # Databases & Storage
    "PostgreSQL", "MySQL", "SQLite", "MongoDB", "Redis", "Cassandra", "DynamoDB",
    "Neo4j", "Pinecone (Vector DB)", "Milvus", "Weaviate", "InfluxDB (Time Series)",
    "Database Indexing", "Query Optimization", "ORM (SQLAlchemy, Prisma, Hibernate)",
    "Database Sharding", "Replication", "ACID Properties", "CAP Theorem", "Change Data Capture (CDC)",

    # DevOps & Cloud
    "Docker", "Kubernetes", "Helm Charts", "AWS", "Google Cloud Platform", "Azure",
    "CI/CD Pipelines", "GitHub Actions", "Jenkins", "GitOps (ArgoCD, Flux)",
    "Terraform", "Infrastructure as Code (Pulumi)", "Nginx", "Envoy Proxy",
    "Load Balancing", "Auto Scaling", "Serverless (Lambda, Cloud Functions)",
    "Service Mesh (Istio)", "Prometheus", "Grafana", "ELK Stack", "OpenTelemetry",

    # Version Control
    "Git", "GitHub", "GitLab", "Bitbucket", "Branching Strategies (Gitflow, Trunk-based)",
    "Merge Conflicts", "Code Reviews", "Conventional Commits",

    # Machine Learning & AI (Core)
    "Linear Regression", "Logistic Regression", "Decision Trees", "Random Forest",
    "Support Vector Machines", "K-Means Clustering", "PCA", "Feature Engineering",
    "Model Evaluation", "Cross Validation", "Bias-Variance Tradeoff", "Hyperparameter Tuning",

    # Deep Learning & Architectures
    "Neural Networks", "Backpropagation", "CNN", "RNN", "LSTM", "GRU", "Transformers",
    "Attention Mechanism", "GANs", "Autoencoders", "Transfer Learning", "Diffusion Models",
    "Graph Neural Networks (GNN)", "Siamese Networks",

    # AI Frameworks & Tools
    "PyTorch", "TensorFlow", "Keras", "Scikit-learn", "XGBoost", "LightGBM", "ONNX",
    "OpenCV", "Hugging Face Transformers", "LangChain", "LlamaIndex", "Haystack",

    # LLMs & Agentic AI
    "Chain-of-Thought (CoT)", "Multi-Agent Systems", "Autonomous Agents", "Prompt Engineering",
    "RAG (Retrieval-Augmented Generation)", "Fine-tuning (LoRA, QLoRA)", "Quantization (GGUF, AWQ)",
    "RLHF", "Function Calling / Tool Use", "AI Guardrails",

    # NLP & Computer Vision
    "Tokenization", "Word Embeddings", "BERT", "GPT Models", "NER", "Text Classification",
    "Sentiment Analysis", "Image Classification", "Object Detection (YOLO, DETR)",
    "Image Segmentation", "Feature Extraction", "OCR",

    # Data & MLOps
    "ETL Pipelines", "Data Warehousing (Snowflake)", "Apache Spark", "Kafka", "Airflow",
    "DVC (Data Version Control)", "MLflow", "Weights & Biases", "Model Deployment (BentoML, Ray Serve)",
    "Feature Stores (Feast)", "Model Monitoring",

    # Security
    "Encryption", "Hashing", "SSL/TLS", "Web Security (OWASP Top 10)", "SQL Injection Prevention",
    "XSS Protection", "OAuth Security", "Zero Trust Architecture", "IAM", "Vault",

    # Testing
    "Unit Testing", "Integration Testing", "E2E Testing (Cypress, Playwright)",
    "Test Driven Development (TDD)", "PyTest", "JUnit", "Mocking", "Load Testing (k6, Locust)",

    # Systems & Networking
    "TCP/IP", "UDP", "QUIC", "DNS", "Sockets Programming", "Network Security",
    "Embedded C", "RTOS", "Linux Kernel Basics", "eBPF", "System Profiling",

    # Advanced & Emerging Tech
    "Blockchain Basics", "Smart Contracts (Solidity)", "Web3.js", "IPFS",
    "Quantum Computing Basics", "Edge Computing", "HPC (CUDA, OpenMP)",

    # Soft Engineering & Business
    "Problem Solving", "Debugging", "Code Optimization", "Scalability Design",
    "Technical Writing", "Project Management", "Agile/Scrum", "Product Analytics",
    "Domain-Driven Design (DDD)", "Financial Engineering Basics"
]
import pdfplumber
import re
import os

def find_skills(text):
    text_lower = text.lower()
    found = []

    for skill in skills:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)

    return found


def resume_extractor(file_path: str) -> dict:
    try:
        if not os.path.exists(file_path):
            return {"error": "File not found"}

        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

        found_skills = find_skills(text)

        return {
            "skills": found_skills,
            "count": len(found_skills)
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    print(resume_extractor("F:\\Important\\CV_ChiranjitSaha.pdf")) 