# TrustLoop
TrustLoop is a two-stage evaluation framework for detecting hallucinations in Retrieval-Augmented Generation (RAG) systems.

Stage 1 uses lightweight pretrained evaluators for:
- Relevance checking
- Groundedness evaluation
- Natural Language Inference (NLI)

Stage 2 uses an LLM verifier for examples that fail or remain uncertain after Stage 1.

The project compares sequential and parallel Stage 1 architectures to study the trade-off between hallucination detection performance, latency, computational cost, and LLM usage.

## Dataset

The project uses the RAGTruth Question Answering subset.

## Project Structure

data/       Dataset files
docs/       Project documentation
models/     Local model files and artifacts
notebooks/  Experiment notebooks
results/    Evaluation outputs and plots
src/        Project source code
