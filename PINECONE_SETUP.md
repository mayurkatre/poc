# Pinecone Integration Guide 🌲

## Overview

Your RAG system now supports **Pinecone** as a production-ready vector database! This guide will help you set it up and migrate from FAISS.

---

## 🎯 Why Pinecone?

- ✅ **Fully Managed** - No infrastructure to maintain
- ✅ **Auto-Scaling** - Handles millions of vectors effortlessly  
- ✅ **Real-time Updates** - Add/update vectors instantly
- ✅ **High Performance** - Sub-second query latency at any scale
- ✅ **Built-in Security** - Enterprise-grade access control

---

## 📋 Prerequisites

1. **Pinecone Account**: Sign up at [https://app.pinecone.io](https://app.pinecone.io)
2. **Free Tier Available**: Yes! Get started with 100K vectors free
3. **API Key**: Available in Pinecone dashboard

---

## 🚀 Quick Start

### Step 1: Get Your Pinecone API Key

1. Go to [Pinecone Dashboard](https://app.pinecone.io)
2. Click on your profile icon → **API Keys**
3. Copy your API key (starts with `pcsk_...`)

### Step 2: Configure Environment

Add Pinecone credentials to your `.env` file:

```bash
# Copy example if you haven't already
cp .env.example .env

# Edit .env and add:
PINECONE_API_KEY=pcsk_your_actual_api_key_here
PINECONE_ENVIRONMENT=us-west-2-gcp
PINECONE_INDEX_NAME=rag-documents
```

### Step 3: Update Configuration

Edit `config/settings.yaml`:

```yaml
vector_store:
  provider: "pinecone"   # Changed from "faiss"
  
  # Pinecone settings
  pinecone_api_key: "${PINECONE_API_KEY}"
  pinecone_environment: "us-west-2-gcp"
  pinecone_index_name: "rag-documents"
  pinecone_namespace: "default"
  
  # Keep these for compatibility
  top_k: 20
```

### Step 4: Create Pinecone Index

The index will be created automatically when you run ingestion, but you can also create it manually:

**Option A: Using Python SDK**
```python
import pinecone

# Initialize
pinecone.init(
    api_key="your-api-key",
    environment="us-west-2-gcp"
)

# Create index
pinecone.create_index(
    name="rag-documents",
    dimension=384,  # Matches all-MiniLM-L6-v2
    metric="cosine"
)
```

**Option B: Using Pinecone Dashboard**
1. Go to **Indexes** in Pinecone dashboard
2. Click **Create Index**
3. Name: `rag-documents`
4. Dimension: `384`
5. Metric: `cosine`
6. Click **Create**

### Step 5: Re-ingest Documents

Since you're switching from FAISS to Pinecone, you need to re-ingest your documents:

```bash
# This will upload all documents to Pinecone
python ingest.py ./documents
```

You should see output like:
```
INFO     | ingestion.embedding_pipeline:add:520 - Pinecone: upserted 112 vectors
SUCCESS  | Ingestion complete!
```

### Step 6: Test It!

Start the application:

```bash
# Backend
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# Frontend (in another terminal)
cd frontend
npm run dev
```

Then query at http://localhost:3000 or test the health endpoint:
```bash
curl http://localhost:8000/health
```

---

## 🔧 Advanced Configuration

### Change Environment

Pinecone offers different environments. Update in `settings.yaml`:

```yaml
pinecone_environment: "us-east-1-aws"  # or other regions
```

Available environments:
- `us-west-2-gcp` (Google Cloud - US West)
- `us-east-1-aws` (AWS - US East)
- `eu-west-1-aws` (AWS - Europe)

### Use Namespaces

Namespaces let you partition data within an index:

```yaml
pinecone_namespace: "production"  # Separate from "staging", "dev", etc.
```

### Adjust Batch Size

For large document sets, tune the upsert batch size in `embedding_pipeline.py`:

```python
# In PineconeVectorStore.add()
batch_size = 100  # Increase for faster uploads (max 1000)
```

---

## 📊 Monitoring & Management

### Check Index Stats

```python
import pinecone

pinecone.init(api_key="your-key", environment="us-west-2-gcp")
index = pinecone.Index("rag-documents")

# Get statistics
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
print(f"Index dimension: {stats['dimension']}")
```

### View in Dashboard

Visit your Pinecone dashboard to see:
- Vector count
- Index usage metrics
- Query logs
- Billing information

### Delete Vectors

To remove specific documents:

```python
# Delete by ID
index.delete(ids=["chunk_001", "chunk_002"])

# Delete entire namespace
index.delete(delete_all=True, namespace="default")
```

---

## 💰 Pricing

**Free Tier** (Starter):
- ✅ 100,000 vectors
- ✅ 1M read units/month
- ✅ Limited to 1 project
- ✅ Community support

**Standard** ($70/month):
- ✅ 1M vectors included
- ✅ $0.04 per 1K additional vectors
- ✅ Higher rate limits
- ✅ Email support

**Enterprise**: Custom pricing
- ✅ Unlimited vectors
- ✅ Dedicated support
- ✅ SLA guarantees

---

## 🔄 Migration from FAISS

### Automatic Migration

When you switch the provider in `settings.yaml` and run `ingest.py`, everything happens automatically:

1. ✅ Embeddings generated (same as before)
2. ✅ Vectors uploaded to Pinecone
3. ✅ Metadata preserved
4. ✅ Ready for queries immediately

### Keep FAISS as Backup

You can use both during migration:

```yaml
# Test Pinecone first
vector_store:
  provider: "pinecone"

# Later switch back if needed
vector_store:
  provider: "faiss"
```

Your FAISS index remains in `.cache/faiss_index/` untouched.

---

## 🐛 Troubleshooting

### Error: "Invalid API Key"

**Solution**: Verify your key in `.env`:
```bash
# Check it's loaded
echo $PINECONE_API_KEY

# Should start with pcsk_
```

### Error: "Index not found"

**Solution**: The index will be created automatically on first ingestion. Or create manually (see Step 4).

### Slow Upload Speeds

**Solution**: Increase batch size in code:
```python
batch_size = 500  # Default is 100
```

### Rate Limit Errors

**Solution**: Add retry logic or reduce concurrency. Pinecone has generous limits but you can add exponential backoff.

---

## 📈 Performance Tips

1. **Batch Your Requests**: Group multiple queries together
2. **Use Appropriate Top-K**: Don't request more results than needed
3. **Enable Metadata Filtering**: Reduces post-processing
4. **Choose Right Index Type**: 
   - `pods` for production (consistent performance)
   - `serverless` for variable workloads (cost-effective)

---

## 🎓 Next Steps

1. ✅ **Test with Small Dataset**: Verify everything works
2. ✅ **Monitor Usage**: Check Pinecone dashboard regularly
3. ✅ **Set Up Alerts**: Configure usage notifications
4. ✅ **Optimize Queries**: Tune top_k and filtering
5. ✅ **Scale Gradually**: Start with free tier, upgrade as needed

---

## 📚 Resources

- [Pinecone Docs](https://docs.pinecone.io)
- [Python SDK Reference](https://docs.pinecone.io/reference/api)
- [Best Practices Guide](https://www.pinecone.io/learn/)
- [Community Forum](https://www.pinecone.io/community/)

---

## 🆘 Support

If you encounter issues:

1. Check Pinecone dashboard for error messages
2. Review application logs
3. Consult Pinecone documentation
4. Ask in community forum
5. Contact Pinecone support (for paid plans)

---

**You're all set! Enjoy scalable, production-ready vector search with Pinecone! 🚀**
