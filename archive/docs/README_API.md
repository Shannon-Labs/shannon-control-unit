# SCU Training API - Implementation Team README

## 🎯 Quick Reference for Engineering Team

**Project:** Shannon Control Unit Training API  
**Goal:** Transform research codebase into production-ready training service  
**Status:** 60% of Phase 1 complete, validation and UX systems needed  
**Timeline:** 2 weeks for critical features, 6-8 weeks total  

---

## 📚 Documentation Files (Start Here)

### 1. **API_STATUS.md** (READ FIRST)
**What:** Current implementation status and gap analysis  
**Why:** Understand what's built vs what needs to be built  
**Key sections:**
- ✅ What's implemented (60% complete)
- ❌ What's missing (40% of Phase 1, all of Phases 2-5)
- 🎯 Priority recommendations (what to build next)
- 🧪 Testing strategy

**Action:** Read before planning sprint

---

### 2. **API_ACTION_PLAN.md** (IMPLEMENT FROM THIS)
**What:** Copy-pasteable implementation tasks  
**Why:** Exact code to write, organized by engineer role  
**Key sections:**
- **Task 1:** Backend Engineer - Configuration Validation
- **Task 2:** ML Engineer - Smart Configuration System
- **Task 3:** Full-Stack Engineer - Python SDK & CLI

**Action:** Engineers start with their assigned task section

---

### 3. **API_DESIGN_PLAN.md** (ARCHITECTURE REFERENCE)
**What:** High-level API design and product vision  
**Why:** Understand the "why" behind implementation decisions  
**Key sections:**
- API architecture vision
- User experience design
- Database schemas
- Implementation roadmap

**Action:** Read for context, reference when architectural decisions needed

---

### 4. **API_IMPLEMENTATION.md** (DEEP TECHNICAL REFERENCE)
**What:** Line-by-line implementation guide (79KB)  
**Why:** Complete technical specification for all components  
**Key sections:**
- Pydantic models
- TrainingEngine class
- Job queue implementation
- FastAPI endpoints
- PostgreSQL setup

**Action:** Reference when building deeper features beyond Week 1

---

## 🏗️ Current Architecture

```
┌─────────────────────────────────────────────────┐
│          USER INTERFACES (to be built)          │
├─────────────────────────────────────────────────┤
│  CLI Tool  │  Python SDK  │  REST API Clients │
└──────┬───────┴──────┬───────┴─────────┬────────┘
       │              │                 │
┌──────▼──────────────▼─────────────────▼────────┐
│           API SERVER (in progress)              │
├─────────────────────────────────────────────────┤
│  • FastAPI server (server.py) ✅                │
│  • Job manager (job_manager.py) ✅              │
│  • Configuration (config.py) ✅                 │
│  • Training engine (training_engine.py) ✅      │
├─────────────────────────────────────────────────┤
│  Missing:                                       │
│  • Configuration validation ❌                  │
│  • Smart auto-config ❌                         │
│  • Database persistence ❌                      │
│  • Authentication ❌                            │
└──────┬──────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────┐
│      CORE TRAINING (already exists)             │
├─────────────────────────────────────────────────┤
│  • SCU controller (shannon_control/control.py)  │
│  • Data utilities (shannon_control/data.py)     │
│  • Metrics calculation (shannon_control/)       │
└─────────────────────────────────────────────────┘
```

---

## 👥 Team Structure (Recommended)

### **Backend Engineer** (Primary Focus: Validation & Persistence)
**Responsibilities:**
- Configuration validation system
- Database integration (SQLite → PostgreSQL)
- API endpoint security
- Error handling & logging

**Primary Files:**
- `scu_api/service/validator.py` (to be created)
- `scu_api/server.py` (add validation endpoints)
- `scu_api/service/job_queue.py` (add persistence)

**Task:** Start with **API_ACTION_PLAN.md Task 1** (Configuration Validation)

---

### **ML Engineer** (Primary Focus: Smart Configuration)
**Responsibilities:**
- Model size detection
- Automatic parameter scaling
- Dataset analysis
- Performance optimization

**Primary Files:**
- `scu_api/service/smart_config.py` (to be created)
- `scu_api/service/model_utils.py` (to be created)

**Task:** Start with **API_ACTION_PLAN.md Task 2** (Smart Configuration)

---

### **Full-Stack Engineer** (Primary Focus: SDK & CLI)
**Responsibilities:**
- Python SDK (sync and async)
- CLI tool
- Documentation
- Example notebooks

**Primary Files:**
- `scu_api/client/sync_client.py` (to be created)
- `scu_api/cli/main.py` (to be created)
- `setup.py` (to be created)

**Task:** Start with **API_ACTION_PLAN.md Task 3** (Python SDK & CLI)

---

### **Team Lead / Architect** (Coordination & Integration)
**Responsibilities:**
- Code review
- Integration testing
- Documentation
- Sprint planning
- Stakeholder updates

**Key Activities:**
- Daily standups (15 min)
- Integration testing (every 2 days)
- Documentation updates (ongoing)
- API_STATUS.md updates (weekly)

---

## 📋 Week 1 Sprint Plan (CRITICAL PRIORITIES)

### **Goal:** Minimum Viable Product (MVP) that real users can use

**Monday AM:** Sprint Kickoff
- [ ] All team members read API_STATUS.md
- [ ] Engineers read their assigned task in API_ACTION_PLAN.md
- [ ] Clarify questions and blockers
- [ ] Set up development environment

**Monday-Tuesday:** Core Features
- **Backend Engineer:** Build configuration validation system
- **ML Engineer:** Build smart configuration system
- **Full-Stack Engineer:** Set up Python SDK structure

**Wednesday-Thursday:** Integration
- **Team:** Daily integration check-ins
- **Backend:** Add validation to job submission
- **ML:** Add auto-config endpoints to server
- **Full-Stack:** Complete CLI tool

**Friday:** Testing & Polish
- [ ] End-to-end test
- [ ] Write unit tests
- [ ] Update documentation
- [ ] Week 1 demo preparation

**Friday PM:** Demo & Retrospective
- [ ] Demo training a model with 3-line command
- [ ] Discuss what worked / what didn't
- [ ] Plan Week 2 priorities

---

## 🎯 Success Criteria (Week 1)

### Technical:
- ✅ Configuration validation catches 80%+ of common errors
- ✅ Auto-configuration works for 10+ model architectures  
- ✅ Python SDK can submit job and monitor progress
- ✅ CLI tool can train model from terminal
- ✅ All paths tested (Unsloth enabled/disabled)
- ✅ Unit tests for all new components

### User-Facing:
- ✅ User can train model with 3 commands:
  ```bash
  scu train --base-model gpt2 --dataset data.txt
  scu status <job-id>
  scu download <job-id>
  ```
- ✅ Configuration validation gives clear error messages
- ✅ Auto-configuration eliminates guesswork
- ✅ Progress bar shows training progress

### Documentation:
- ✅ API_ACTION_PLAN.md updated if needed
- ✅ New endpoints documented
- ✅ SDK usage examples written
- ✅ Troubleshooting guide started

---

## 🚀 Quick Start Commands

### Setup Development Environment:

```bash
cd /Volumes/VIXinSSD/shannon-control-unit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .[dev,server]

# Verify installation
scu --help
python -c "from scu_api import SCUClient; print('✅ SDK loaded')"
```

### Test Current Implementation:

```bash
# Start API server (in one terminal)
python -m scu_api.server

# Test health endpoint (in another terminal)
curl http://localhost:8000/jobs
# Should return: {"jobs": {}}

# Test with tiny model
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "sshleifer/tiny-gpt2",
    "train_data": "data/train.txt",
    "steps": 5,
    "batch_size": 1,
    "lora_r": 2
  }'

# Check status (copy job_id from response)
curl http://localhost:8000/jobs/<job-id>
```

### Test With CLI (after SDK is built):

```bash
# Test health
scu health

# Train tiny model (will take ~1 minute)
scu train \
  --base-model sshleifer/tiny-gpt2 \
  --train-data data/train.txt \
  --steps 5 \
  --lora-r 2 \
  --wait

# Should show progress bar and complete successfully

# Auto-config then train
scu train --base-model gpt2 --train-data data/train.txt --auto-config --wait

# Check status or list jobs
scu status <job-id>
scu jobs

# Download artifact
scu download <job-id> --output adapters/
```

---

## 🔍 Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'scu_api'`
```bash
# Solution: Install in development mode
pip install -e .
```

**Problem:** `CUDA out of memory` during training
```bash
# Solution 1: Reduce batch_size
curl ... -d '{"batch_size": 1, ...}'

# Solution 2: Reduce lora_r
curl ... -d '{"lora_r": 4, ...}'

# Solution 3: Enable 4-bit (if CUDA available)
curl ... -d '{"use_unsloth": true}'
```

**Problem:** API server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000
# Kill process if needed: kill -9 <PID>

# Or use different port
python -m scu_api.server --port 8001
```

**Problem:** Unsloth import error
```bash
# Install unsloth (optional, not required)
pip install unsloth

# Or disable unsloth
curl ... -d '{"use_unsloth": false}'
```

---

## 📊 Key Performance Metrics to Track

### Training Efficiency:
- **Time to first step:** Should be < 30 seconds
- **Tokens/second:** Track per model size (gpt2 ~100 tok/s, llama3b ~500 tok/s with Unsloth)
- **Memory usage:** Should stay under GPU capacity
- **BPT improvement:** Should match research baselines (±5%)

### API Performance:
- **Job submission:** < 1 second
- **Status query:** < 100ms
- **Validation:** < 2 seconds
- **Auto-config:** < 500ms

### User Success Metrics:
- **Job success rate:** Target 90%+
- **Validation catch rate:** Target 85% of errors caught
- **Auto-config usage:** Track how many users use vs manual config
- **Support tickets:** Track questions/issues per user

---

## 🔐 Security & Production Readiness

**Before production deployment:**

- [ ] Add API key authentication
- [ ] Rate limiting (jobs per user per hour)
- [ ] Model whitelist for security
- [ ] Input sanitization
- [ ] Docker image security scan
- [ ] HTTPS/TLS termination
- [ ] PostgreSQL instead of SQLite
- [ ] Persistent volume for adapters
- [ ] Logging aggregation
- [ ] Error tracking (Sentry)

**Not needed for Week 1 MVP** (focus on functionality first)

---

## 🔄 Integration Points

### With Existing SCU Codebase:
```
scu_api/training_engine.py → imports → shannon_control/control.py
                                          shannon_control/data.py
                                          shannon_control/metrics.py
                                          
scu_api/config.py → uses → TrainingConfig (Pydantic)

FastAPI server → calls → JobManager → creates → TrainingEngine
                                     ↓
                                  asyncio task
```

All imports should "just work" if shannon_control package is installed.

---

## 📅 Week 2+ Preview

**After Week 1 MVP is complete:**

**Week 2:**
- SQLite persistence layer
- Job cancellation API
- Hugging Face Hub auto-upload
- Test coverage to 80%

**Week 3-4:**
- PostgreSQL migration
- Authentication system
- Multi-user support
- Monitoring & logging

**Week 5-6:**
- Advanced features (checkpoint resume, dataset caching)
- Performance optimization
- Load testing
- Documentation

**Week 7-8:**
- Production deployment
- User onboarding
- Community examples
- Launch!

---

## 📞 Getting Help

**Before asking for help:**
1. Check API_STATUS.md for current status
2. Check API_ACTION_PLAN.md for implementation details
3. Check existing code in scu_api/ directory
4. Run quick tests to reproduce issue

**When asking for help, provide:**
- What you were trying to do
- What happened (error messages, logs)
- What you've already tried
- Your environment (OS, Python version, GPU?)

**Communication:**
- Daily standups: 9:00 AM (15 minutes)
- Slack/Discord for quick questions
- GitHub issues for bugs
- GitHub PRs for code review

---

## 🎓 Learning Resources

**Understanding SCU Core:**
- `SCU_Technical_Report_v1.pdf` - Technical paper
- `shannon_control/control.py` - PI controller implementation
- `scripts/train_scu.py` - Original training script (reference)

**Understanding FastAPI:**
- FastAPI documentation: https://fastapi.tiangolo.com
- Pydantic models: https://docs.pydantic.dev
- `scu_api/server.py` - Existing endpoints

**Understanding Unsloth:**
- `scu_api/training_engine.py` lines 228-250 - Unsloth path
- `Dockerfile.unsloth` - Container setup
- Note: Unsloth is optional (API works without it)

---

## 🏆 Definition of Done (Week 1)

A task is "done" when:

- ✅ Code is written and committed
- ✅ Unit tests pass (minimum 80% coverage)
- ✅ Integration test passes end-to-end
- ✅ Documentation is updated
- ✅ Code reviewed by at least one other engineer
- ✅ Deploys successfully in Docker
- ✅ Demo works for a new user

---

## 🎯 Final Thoughts

**You're building something important:**
- Democratizes LLM fine-tuning (no PhD required)
- Improves model quality (6-12% BPT improvements)
- Saves compute costs (optimal regularization = less training time)

**The team has done excellent work so far:**
- Clean architecture
- Good separation of concerns
- Smart use of Unsloth for performance
- Solid foundation to build on

**Week 1 success will unlock:**
- Real users trying the API
- Feedback for improvement
- Momentum for the project
- Foundation for production launch

**Let's make this happen! 🚀**
