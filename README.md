# 🤖 GAIA Agent

GAIA is a general AI agent framework designed to answer GAIA benchmark data ste questions (Level 1). It’s built to support flexible and interactive multi-step workflows using LLMs, with environment-configurable access to APIs and tools. This is a HuggingFace Agent Course final project.

---

## 🚀 Features

- Dynamically selects and uses tools
- Integrates with LLMs via OpenAI
- Deployed on Gradio app. 
---

## 🛠️ Project Structure

```
GAIA_agent/
├── app.py                # Entry point for running the agent
├── agent/
│   ├── agent.py          # Core agent logic and state management
│   ├── tool_used.py      # Tool usage node and external tool handlers
├── requirements.txt      # Python dependencies
├── .env.example          # Example environment configuration
└── README.md             # This file
```

---

## 🔐 Environment Setup

You need to create a `.env` file in the root directory with your API keys and environment settings.

### ✅ Step 1: Create your `.env` file

Copy the sample:
```bash
cp .env.example .env
```

### ✅ Step 2: Add your API keys

Open `.env` and update the placeholder values. Example:

```env
OPENAI_API_KEY=your-openai-key
HUGGINGFACE_API_KEY=your-huggingface-key
TAVILY_API_KEY=your-tavilyapi-key
```

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Peyman7/GAIA_agent.git
   cd GAIA_agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧠 Running the Agent

You can run the agent with:

```bash
python app.py
```

The agent will prompt for user input and begin reasoning using the defined node flow.

---

## 📝 License

This project is licensed under the MIT License.

---

## 📬 Contributing

Pull requests and ideas are welcome! If you encounter any bugs or have suggestions, feel free to open an issue.
