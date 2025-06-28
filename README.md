# Multi-Agent RAG System 🤖

A comprehensive, web-based RAG (Retrieval-Augmented Generation) system with an intuitive Streamlit interface that combines document processing, web scraping, and advanced language model capabilities to provide intelligent answers, summaries, and evaluations.

![system capabilities](https://github.com/user-attachments/assets/365a7dd5-8131-427a-870c-1cf3a9ab7499)


## 🎯 What is this project about?

This advanced Multi-Agent RAG system provides a complete web application that enables you to:

- **📄 Process PDF Documents** - Upload and analyze PDF files with intelligent text extraction
- **🌐 Web Content Analysis** - Scrape and process content from any URL
- **❓ Intelligent Q&A** - Ask questions and get accurate, context-aware answers
- **📊 Advanced Summarization** - Generate multiple types of summaries (abstractive, extractive, key points)
- **📈 Comprehensive Evaluation** - Quality assessment with 7+ advanced metrics
- **🔍 Semantic Search** - Find relevant content using vector similarity search
- **⚡ Real-time Processing** - Progress tracking and status updates

The system features a user-friendly Streamlit web interface that makes advanced RAG capabilities accessible to everyone, from researchers to business professionals.

## 🌟 Key Features

### 🖥️ Web Interface
- **Streamlit-powered UI** - Clean, intuitive interface for all operations
- **Real-time Progress Tracking** - Visual progress bars and status updates
- **Tabbed Interface** - Organized sections for Q&A, summaries, evaluation, and document info
- **Quick Question Suggestions** - Pre-defined questions to get started quickly
- **Responsive Design** - Works on desktop and mobile devices

### 📄 Document Processing
- **PDF Upload Support** - Drag-and-drop PDF processing with validation
- **Text Extraction** - Advanced PDF text extraction with error handling
- **Document Validation** - Pre-processing checks and warnings
- **Metadata Extraction** - File size, pages, encryption status, author info
- **Processing Recommendations** - Intelligent suggestions for optimization

### 🌐 Web Scraping Integration
- **URL Processing** - Extract and analyze content from any website
- **Content Validation** - Ensure sufficient content for analysis
- **Error Handling** - Robust error management for web scraping
- **Progress Tracking** - Real-time updates during URL processing

### 🤖 Advanced AI Capabilities
- **Multi-Agent Architecture** - Specialized components for different tasks
- **Intelligent Chunking** - Smart text segmentation for optimal processing
- **Vector Embeddings** - High-quality semantic representations
- **Context-Aware Answers** - Responses that understand document context
- **Quality Evaluation** - Comprehensive answer assessment metrics

### 📊 Evaluation & Metrics
- **Relevance Scoring** - How well answers match questions
- **Confidence Assessment** - Reliability measurement
- **Quality Metrics** - Overall response quality evaluation
- **System Performance** - Processing time and resource usage tracking
- **Source Attribution** - Clear references to original content

## 🏗️ System Architecture

### Complete Data Flow Architecture
 
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Documents     │───▶│   Text Chunks    │───▶│   Embeddings   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
         ┌─────────────────┐    ┌──────────────────┐    │
         │   Web URLs      │───▶│   Web Scraping   │────┤
         │                 │    │   & Processing   │    │
         └─────────────────┘    └──────────────────┘    │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Vector Search   │◀───│  FAISS Index    │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │                       ▼
         │              ┌─────────────────┐
         │              │ Retrieved Chunks│
         │              │                 │
         │              └─────────────────┘
         │                       │
         └───────────────────────┼─────────────────────────┐
                                 ▼                         │
                        ┌─────────────────┐                │
                        │  Summarization  │                │
                        │                 │                │
                        └─────────────────┘                │
                                 │                         │
                                 ▼                         ▼
                        ┌─────────────────┐    ┌─────────────────┐
                        │ Context Summary │───▶│  Gemini LLM     │
                        │                 │    │   Generation    │
                        └─────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                        ┌─────────────────────────────────────────┐
                        │           Final Answer                  │
                        └─────────────────────────────────────────┘
                                                         │
                                                         ▼
                        ┌────────────────────────────────────────┐
                        │        Advanced Evaluation Metrics     │
                        │  • Relevance Score                     │
                        │  • Context Utilization                 │
                        │  • Completeness Score                  │
                        │  • Factual Consistency                 │
                        │  • Length Appropriateness              │
                        │  • Confidence Score                    │
                        │  • Overall Quality Score               │
                        └────────────────────────────────────────┘
```

### Web Interface Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                    Streamlit Web Interface                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐│
│  │    Q&A      │ │  Summary    │ │ Evaluation  │ │ Doc Info    ││
│  │    Tab      │ │    Tab      │ │    Tab      │ │    Tab      ││
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                Multi-Agent RAG System                          │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐│
│  │   Document      │    │   Web Scraper    │    │   Vector    ││
│  │   Processor     │    │                  │    │   Store     ││
│  └─────────────────┘    └──────────────────┘    └─────────────┘│
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────┐│
│  │   LLM Handler   │    │   Summarizer     │    │ Evaluation  ││
│  │   (Gemini)      │    │                  │    │  Metrics    ││
│  └─────────────────┘    └──────────────────┘    └─────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Melina-Singh/MultiAgent_RAG.git


# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

### 3. Run the Application

```bash
streamlit run main_app.py
```

The application will open in your browser at `http://localhost:8501`.

## 📦 Required Dependencies

```txt
streamlit>=1.28.0
google-generativeai>=0.3.0
sentence-transformers>=2.2.0
transformers>=4.30.0
torch>=2.0.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
scikit-learn>=1.3.0
numpy>=1.24.0
python-dotenv>=1.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
urllib3>=2.0.0
PyPDF2>=3.0.0  # for PDF processing
pathlib
tempfile
```

## 🎮 How to Use

### 1. **Upload Content**
   - **PDF Upload**: Use the sidebar to upload PDF documents
   - **URL Processing**: Enter any website URL to scrape content
   - **Progress Tracking**: Watch real-time processing updates

### 2. **Ask Questions**
   - Navigate to the **Q&A tab**
   - Use quick question suggestions or type custom questions
   - Get intelligent answers with quality metrics
   - View source chunks that informed the answer

### 3. **Generate Summaries**
   - Go to the **Summary tab**
   - View automatic summaries of processed content
   - Choose from different summary types (abstractive, extractive, key points)
   - Generate custom summaries as needed

### 4. **Evaluate Performance**
   - Check the **Evaluation tab**
   - View system performance metrics
   - Run comprehensive evaluation tests
   - Monitor accuracy and resource usage

### 5. **Review Document Info**
   - Access the **Document Info tab**
   - View metadata and processing details
   - Check for warnings and recommendations
   - Review document statistics

## 🔧 Backend Components

### 1. `main_app.py` - Streamlit Interface
- **MultiAgentRAGSystem Class** - Main orchestrator for all operations
- **Progress Tracking** - Real-time status updates during processing
- **Error Handling** - Comprehensive error management and user feedback
- **Session Management** - Maintains state across user interactions
- **Responsive UI** - Clean, organized interface with tabs and metrics

### 2. `document_processor.py` - Document Processing
- **PDF Text Extraction** - Robust PDF processing with validation
- **Text Chunking** - Intelligent text segmentation
- **Metadata Extraction** - Document properties and statistics
- **Processing Status** - Validation and recommendation system

### 3. `vector_store.py` - Vector Database
- **FAISS Integration** - High-performance similarity search
- **Embedding Management** - Vector storage and retrieval
- **Similarity Search** - Semantic content matching
- **Batch Operations** - Efficient bulk processing

### 4. `llm_handler.py` - Language Model Interface
- **Google Gemini Integration** - Advanced text generation
- **Answer Generation** - Context-aware response creation
- **Quality Control** - Response validation and optimization

### 5. `evaluation_metrics.py` - Quality Assessment
- **Multi-Metric Evaluation** - Comprehensive answer assessment
- **Relevance Scoring** - Question-answer alignment measurement
- **Confidence Metrics** - Reliability assessment
- **Performance Tracking** - System efficiency monitoring

### 6. `summarizer.py` - Text Summarization
- **Multiple Summary Types** - Abstractive, extractive, and key points
- **Context-Aware Summarization** - Query-focused summaries
- **Fallback Mechanisms** - Robust summarization with error handling

### 7. `web_scraper.py` - Web Content Extraction
- **URL Processing** - Intelligent web content extraction
- **Content Validation** - Quality checks for scraped content
- **Error Handling** - Robust web scraping with fallbacks

## 📊 Evaluation Metrics

The system provides comprehensive evaluation with these metrics:

- **Relevance Score** (0-1): How well the answer addresses the question
- **Confidence Score** (0-1): System confidence in the answer quality
- **Quality Score** (0-1): Overall answer quality assessment
- **Context Utilization**: Effectiveness of using retrieved context
- **Completeness**: Whether all aspects of the question are addressed
- **Factual Consistency**: Accuracy compared to source material

## 💡 Use Cases


### 1. **Document Analysis**
- Process business reports, manuals, or legal documents
- Extract key information and insights
- Generate executive summaries

### 2. **Web Content Research**
- Analyze articles, blog posts, or news content
- Compare information across multiple sources
- Generate summaries of web-based research

## 🔍 Troubleshooting

### Common Issues

**Application Won't Start**
```bash
# Check Streamlit installation
pip install --upgrade streamlit

# Verify Python version (3.9+)
python --version

# Run with verbose output
streamlit run main_app.py --logger.level=debug
```

**API Key Issues**
```bash
# Verify API key is set
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', 'Set' if os.getenv('GOOGLE_API_KEY') else 'Not Set')"
```

**PDF Processing Errors**
- Ensure PDF is not corrupted or password-protected
- Check file size (recommended < 10MB for optimal performance)
- Try a different PDF if issues persist

**Memory Issues**
- Reduce the number of chunks processed simultaneously
- Use CPU version of FAISS for lower memory usage
- Process documents in smaller batches

**Web Scraping Failures**
- Verify URL is accessible and not behind authentication
- Check internet connection
- Some websites may block automated scraping


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the excellent web framework
- **Google Gemini** for powerful language generation
- **FAISS** for efficient vector similarity search
- **Sentence Transformers** for high-quality embeddings
- **Hugging Face** for NLP model ecosystem




