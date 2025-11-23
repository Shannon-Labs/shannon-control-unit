# SCU Training API - Technical Implementation Guide

## For the Development Team: Code-Ready Architecture

This document provides line-by-line implementation guidance for building the SCU Training API service. It's designed for an experienced AI/ML engineer team to execute directly.

---

## Core Architecture Stack

**Backend Framework:** FastAPI + Pydantic v2
**Database:** PostgreSQL (dev: SQLite)
**Async Runtime:** asyncio + aiofiles
**ML Stack:** PyTorch + Transformers + PEFT + Accelerate
**GPU Management:** pynvml (NVIDIA Management Library)
**Task Queue:** asyncio Queue (custom implementation)
**Storage:** Local filesystem + HuggingFace Hub

---

## Phase 1: Core API Foundation

### Step 1.1: Repository Structure

```
scu_api/
├── __init__.py
├── main.py                 # FastAPI app entry point
├── config.py              # Pydantic config models
├── job.py                 # Job lifecycle management
├── types.py               # Data models and schemas
├── service/
│   ├── __init__.py
│   ├── training_engine.py # Core training logic
│   ├── job_queue.py       # Job scheduling
│   └── storage.py         # Adapter storage
├── client/
│   ├── __init__.py
│   └── sync_client.py     # Synchronous Python client
├── cli/
│   ├── __init__.py
│   └── main.py            # Click-based CLI
└── tests/
    ├── __init__.py
    ├── test_training_engine.py
    └── test_job_queue.py
```

### Step 1.2: Key Data Models (types.py)

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path

class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TrainingPhase(str, Enum):
    LOADING_MODEL = "loading_model"
    PREPARING_DATA = "preparing_data"
    TRAINING = "training"
    SAVING_ADAPTER = "saving_adapter"
    UPLOADING = "uploading"

class ControlConfig(BaseModel):
    """SCU controller configuration."""
    target_s: float = Field(0.01, ge=0.001, le=0.5, description="Target information ratio")
    Kp: float = Field(0.8, ge=0.0, le=10.0, description="Proportional gain")
    Ki: float = Field(0.15, ge=0.0, le=10.0, description="Integral gain")
    deadband: float = Field(0.002, ge=0.0, le=0.1, description="Control deadband")
    lambda_init: float = Field(1.0, ge=1e-6, le=100.0, description="Initial lambda")
    lambda_min: float = Field(1e-4, ge=1e-8, le=1.0, description="Minimum lambda")
    lambda_max: float = Field(10.0, ge=0.1, le=1000.0, description="Maximum lambda")

class LoRAConfig(BaseModel):
    """LoRA configuration with defaults per model size."""
    r: int = Field(16, ge=1, le=128, description="LoRA rank")
    alpha: int = Field(32, ge=1, le=256, description="LoRA alpha")
    dropout: float = Field(0.05, ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"],
        description="Target modules for LoRA"
    )

class TrainingConfig(BaseModel):
    """Complete training configuration."""
    base_model: str = Field(..., description="HuggingFace model ID")
    train_data_path: str = Field(..., description="Path/URL to training data")
    val_data_path: Optional[str] = Field(None, description="Validation data path")
    
    # Control
    control: ControlConfig = Field(default_factory=ControlConfig)
    
    # LoRA
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    
    # Training hyperparams
    prior_sigma: float = Field(0.01, ge=1e-4, le=1.0, description="Prior std dev")
    epochs: Optional[int] = Field(None, ge=1, description="Number of epochs")
    steps: Optional[int] = Field(None, ge=1, description="Number of steps (overrides epochs)")
    batch_size: int = Field(4, ge=1, le=256, description="Batch size")
    learning_rate: float = Field(5e-5, ge=1e-7, le=1.0, description="Learning rate")
    block_size: int = Field(4096, ge=128, le=32768, description="Sequence length")
    gradient_accumulation_steps: int = Field(1, ge=1, le=100, description="Gradient accumulation")
    
    # Output
    adapter_output_dir: str = Field(..., description="Output directory for adapter")
    auto_upload_to_hub: bool = Field(True, description="Upload to HF Hub when complete")
    hub_adapter_name: Optional[str] = Field(None, description="Name on HF Hub")
    
    # Hardware
    use_4bit: bool = Field(True, description="Use 4-bit quantization (CUDA only)")
    use_fp16: bool = Field(True, description="Use FP16 mixed precision")
    gpu_id: Optional[int] = Field(None, description="Specific GPU to use (None = auto)")

class TrainingMetrics(BaseModel):
    """Real-time training metrics for API responses."""
    step: int
    epoch: Optional[float]
    data_bpt: float
    param_bpt: float
    total_bpt: float
    s_ratio: float
    lambda_value: float
    learning_rate: float
    loss: float
    tokens_per_second: float
    eta_minutes: float
    timestamp: datetime = Field(default_factory=datetime.now)

class JobMetadata(BaseModel):
    """Job metadata for database storage."""
    job_id: str
    base_model: str
    config: TrainingConfig
    status: JobStatus
    priority: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    current_phase: Optional[TrainingPhase] = None
    progress_percent: float = 0.0
    adapter_path: Optional[str] = None

class AdapterArtifact(BaseModel):
    """Trained adapter metadata."""
    job_id: str
    base_model: str
    local_path: Path
    hub_model_id: Optional[str] = None
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: datetime
    files: List[str] = Field(default_factory=list)
```

### Step 1.3: Training Engine Refactor (service/training_engine.py)

**Extract from current `train_scu.py` and refactor:**

```python
import asyncio
import torch
import json
import time
import logging
from pathlib import Path
from typing import Callable, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from datetime import datetime

from ..types import TrainingConfig, TrainingMetrics, TrainingPhase
from .. import control


class TrainingEngine:
    """
    Core training engine that executes SCU training.
    Decoupled from API layer for testability.
    """
    
    def __init__(self, job_id: str, gpu_id: Optional[int] = None):
        self.job_id = job_id
        self.gpu_id = gpu_id
        self.logger = logging.getLogger(f"TrainingEngine.{job_id}")
        self.accelerator = None
        self.model = None
        self.tokenizer = None
        self.controller_state = {
            'lambda': None,
            'I': 0.0,
            'S_hat': None
        }
    
    async def run_training(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[TrainingMetrics], None]] = None
    ) -> AdapterArtifact:
        """
        Execute full training pipeline with SCU control.
        
        This method orchestrates:
        1. Model loading and quantization
        2. LoRA configuration
        3. Data preprocessing
        4. Training loop with SCU control
        5. Checkpoint saving
        6. Upload to Hub (if enabled)
        
        Args:
            config: Validated training configuration
            progress_callback: Called every N steps with metrics
            
        Returns:
            AdapterArtifact with paths and metadata
        """
        
        # Phase 1: Setup
        await self._setup_environment(config)
        
        # Phase 2: Load model
        await self._update_phase(TrainingPhase.LOADING_MODEL)
        self._load_model_and_tokenizer(config)
        
        # Phase 3: Prepare data
        await self._update_phase(TrainingPhase.PREPARING_DATA)
        train_chunks = await self._prepare_data(config)
        
        # Phase 4: Training
        await self._update_phase(TrainingPhase.TRAINING)
        final_metrics = self._training_loop(
            config, train_chunks, progress_callback
        )
        
        # Phase 5: Save adapter
        await self._update_phase(TrainingPhase.SAVING_ADAPTER)
        adapter_path = self._save_adapter(config)
        
        # Phase 6: Upload to Hub
        if config.auto_upload_to_hub:
            await self._update_phase(TrainingPhase.UPLOADING)
            hub_model_id = await self._upload_to_hub(config, adapter_path)
        else:
            hub_model_id = None
        
        # Build artifact
        artifact = AdapterArtifact(
            job_id=self.job_id,
            base_model=config.base_model,
            local_path=adapter_path,
            hub_model_id=hub_model_id,
            config=config.dict(),
            metrics=final_metrics,
            created_at=datetime.now(),
            files=[f.name for f in adapter_path.glob("*") if f.is_file()]
        )
        
        return artifact
    
    def _setup_environment(self, config: TrainingConfig):
        """Configure PyTorch, CUDA, and random seeds."""
        
        # Set GPU (if specified)
        if self.gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.gpu_id)
            self.logger.info(f"Using GPU {self.gpu_id}")
        
        # Set random seeds
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
    
    def _load_model_and_tokenizer(self, config: TrainingConfig):
        """Load and configure model with LoRA."""
        
        # Auto-detect device capabilities
        device_map = self._get_device_map(config)
        torch_dtype = self._get_torch_dtype(config)
        
        # Create quantization config (CUDA only)
        quantization_config = None
        if config.use_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        
        # Load model
        self.logger.info(f"Loading model: {config.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True
        )
        
        # Enable memory optimizations
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            self.logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.r,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            modules_to_save=None,
            bias="none",
            inference_mode=False
        )
        
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()
        
        # Initialize control state
        self.controller_state['lambda'] = config.control.lambda_init
    
    async def _prepare_data(self, config: TrainingConfig) -> list:
        """Load and tokenize training data."""
        
        from .. import data
        
        # Load raw texts
        self.logger.info(f"Loading data from {config.train_data_path}")
        texts = data.load_texts_from_file(config.train_data_path)
        
        # Tokenize and chunk
        block_size = config.block_size
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'max_position_embeddings'):
            block_size = min(block_size, self.model.config.max_position_embeddings)
        
        self.logger.info(f"Tokenizing with block_size={block_size}")
        chunks = data.tokenize_and_chunk(
            texts, self.tokenizer, block_size=block_size, shuffle=True
        )
        
        self.logger.info(f"Created {len(chunks)} training chunks")
        return chunks
    
    def _training_loop(
        self,
        config: TrainingConfig,
        train_chunks: list,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """Main training loop with SCU control integration."""
        
        import torch.optim as optim
        from transformers import get_linear_schedule_with_warmup
        
        # Setup optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0  # Important: must be 0 for SCU
        )
        
        # Calculate training steps
        if config.steps:
            num_training_steps = config.steps
        else:
            steps_per_epoch = len(train_chunks) // config.batch_size
            num_training_steps = config.epochs * steps_per_epoch
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        # Setup accelerator
        self.accelerator = Accelerator(
            mixed_precision="fp16" if config.use_fp16 and torch.cuda.is_available() else None,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            device_placement=True
        )
        
        self.model, optimizer, scheduler = self.accelerator.prepare(
            self.model, optimizer, scheduler
        )
        
        # Create data iterator
        data_iterator = data.create_data_iterator(train_chunks, config.batch_size)
        
        # Training metrics tracking
        tokens_per_epoch = len(train_chunks) * config.block_size
        global_step = 0
        start_time = time.time()
        
        # Metrics log file
        metrics_log_path = Path(config.adapter_output_dir) / "training_metrics.csv"
        metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metrics_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'data_bpt', 'param_bpt', 's_ratio', 
                           'lambda', 'loss', 'lr', 'tokens_per_sec'])
            
            for step in range(num_training_steps):
                global_step = step
                
                # Get batch
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    data_iterator = data.create_data_iterator(train_chunks, config.batch_size)
                    batch = next(data_iterator)
                
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    batch_ids = torch.tensor([c['input_ids'] for c in batch])
                    batch_mask = torch.tensor([c['attention_mask'] for c in batch])
                    
                    batch_ids = batch_ids.to(self.accelerator.device)
                    batch_mask = batch_mask.to(self.accelerator.device)
                    
                    outputs = self.model(
                        input_ids=batch_ids,
                        attention_mask=batch_mask,
                        labels=batch_ids.clone()
                    )
                    
                    # Calculate BPT metrics
                    data_bpt = control.calculate_data_bpt(outputs.loss.item())
                    param_bpt = control.calculate_param_bpt(
                        self.model, 
                        sigma=config.prior_sigma,
                        tokens_per_epoch=tokens_per_epoch
                    )
                    s_ratio = control.calculate_s_ratio(data_bpt, param_bpt)
                    
                    # Update controller
                    lmbda, I, S_hat = control.update_lambda(
                        self.controller_state['lambda'],
                        s_ratio,
                        config.control.target_s,
                        self.controller_state['I'],
                        Kp=config.control.Kp,
                        Ki=config.control.Ki,
                        deadband=config.control.deadband,
                        lmin=config.control.lambda_min,
                        lmax=config.control.lambda_max,
                        S_hat=self.controller_state['S_hat']
                    )
                    
                    self.controller_state.update({'lambda': lmbda, 'I': I, 'S_hat': S_hat})
                    
                    # Total loss with regularization
                    reg_loss = param_bpt * math.log(2) * tokens_per_epoch
                    total_loss = outputs.loss + lmbda * reg_loss
                    
                    # Backward pass
                    self.accelerator.backward(total_loss)
                    
                    if self.accelerator.sync_gradients:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                
                # Log metrics
                if step % 10 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = (step * config.batch_size * config.block_size) / elapsed
                    lr = scheduler.get_last_lr()[0]
                    
                    self.logger.info(
                        f"Step {step}: DataBPT={data_bpt:.3f}, ParamBPT={param_bpt:.4f}, "
                        f"S={s_ratio:.3f}, λ={lmbda:.4f}, Loss={total_loss.item():.3f}"
                    )
                    
                    # Write CSV
                    writer.writerow([
                        step, f"{data_bpt:.4f}", f"{param_bpt:.6f}",
                        f"{s_ratio:.6f}", f"{lmbda:.6f}",
                        f"{total_loss.item():.4f}", f"{lr:.8f}",
                        f"{tokens_per_sec:.2f}"
                    ])
                    f.flush()
                    
                    # Callback for progress updates
                    if progress_callback:
                        metrics = TrainingMetrics(
                            step=step,
                            data_bpt=data_bpt,
                            param_bpt=param_bpt,
                            total_bpt=data_bpt + param_bpt,
                            s_ratio=s_ratio,
                            lambda_value=lmbda,
                            learning_rate=lr,
                            loss=total_loss.item(),
                            tokens_per_second=tokens_per_sec,
                            eta_minutes=(num_training_steps - step) / (tokens_per_sec / 60)
                        )
                        progress_callback(metrics)
        
        # Return final metrics
        return {
            'final_step': global_step,
            'final_data_bpt': data_bpt,
            'final_param_bpt': param_bpt,
            'final_s_ratio': s_ratio,
            'final_lambda': lmbda,
            'final_loss': total_loss.item(),
            'training_time_seconds': time.time() - start_time,
            'metrics_log_path': str(metrics_log_path)
        }
    
    def _save_adapter(self, config: TrainingConfig) -> Path:
        """Save trained adapter to disk."""
        
        output_dir = Path(config.adapter_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving adapter to {output_dir}")
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training config
        with open(output_dir / "training_config.json", 'w') as f:
            json.dump(config.dict(), f, indent=2, default=str)
        
        return output_dir
    
    def _upload_to_hub(self, config: TrainingConfig, adapter_path: Path) -> str:
        """Upload adapter to Hugging Face Hub."""
        
        from huggingface_hub import HfApi, create_repo, get_full_repo_name
        
        api = HfApi()
        
        # Create repo name
        if config.hub_adapter_name:
            repo_id = config.hub_adapter_name
        else:
            repo_id = f"scu-{config.base_model.split('/')[-1]}-{int(time.time())}"
        
        full_repo_id = get_full_repo_name(repo_id)
        
        self.logger.info(f"Uploading to Hub: {full_repo_id}")
        
        # Create repo (if doesn't exist)
        try:
            create_repo(full_repo_id, private=False, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Could not create repo: {e}")
            return full_repo_id
        
        # Upload files
        api.upload_folder(
            folder_path=str(adapter_path),
            repo_id=full_repo_id,
            repo_type="model"
        )
        
        self.logger.info(f"Upload complete: https://huggingface.co/{full_repo_id}")
        return full_repo_id
    
    async def _update_phase(self, phase: TrainingPhase):
        """Update current training phase (for progress tracking)."""
        self.logger.info(f"Phase: {phase.value}")
        # This would update job status in database
        # For now, just log


# Helper methods

def _get_device_map(config: TrainingConfig) -> Optional[dict]:
    """Determine device map based on hardware."""
    
    if torch.cuda.is_available():
        return "auto"  # Let accelerate manage
    elif torch.backends.mps.is_available():
        return {"": "mps"}
    else:
        return {"": "cpu"}


def _get_torch_dtype(config: TrainingConfig) -> torch.dtype:
    """Get appropriate torch dtype."""
    
    if torch.cuda.is_available():
        return torch.float16 if config.use_fp16 else torch.float32
    elif torch.backends.mps.is_available():
        return torch.float32  # MPS doesn't support FP16 well yet
    else:
        return torch.float32
```

### Step 1.4: Job Queue Implementation (service/job_queue.py)

```python
import asyncio
import json
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
import logging

from ..types import JobStatus, JobMetadata, TrainingConfig
from .training_engine import TrainingEngine


class JobQueueManager:
    """
    Manages training job queue with GPU resource allocation.
    Supports priorities and concurrent job execution.
    """
    
    def __init__(self, db_path: Path, max_concurrent_jobs: int = 2):
        self.db_path = db_path
        self.max_concurrent_jobs = max_concurrent_jobs
        self.queue = asyncio.PriorityQueue()
        self.active_jobs: Dict[str, asyncio.Task] = {}
        self.logger = logging.getLogger("JobQueueManager")
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Setup SQLite database for job persistence."""
        
        import sqlite3
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                base_model TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                progress_percent REAL DEFAULT 0.0,
                priority INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                adapter_path TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                data_bpt REAL,
                param_bpt REAL,
                s_ratio REAL,
                lambda_value REAL,
                loss REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (job_id) REFERENCES jobs(job_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def submit_job(
        self,
        config: TrainingConfig,
        priority: int = 0
    ) -> str:
        """Submit a new training job to the queue."""
        
        import uuid
        
        job_id = str(uuid.uuid4())[:8]  # Short ID
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO jobs (job_id, base_model, config_json, status, priority)
            VALUES (?, ?, ?, ?, ?)
        """, (job_id, config.base_model, json.dumps(config.dict()), JobStatus.QUEUED.value, priority))
        
        conn.commit()
        conn.close()
        
        # Add to in-memory queue
        await self.queue.put((-priority, job_id, config))
        
        self.logger.info(f"Job {job_id} queued (priority {priority})")
        return job_id
    
    async def start_queue_processor(self):
        """Start processing jobs from queue."""
        
        self.logger.info("Starting job queue processor")
        
        while True:
            # Wait for available GPU slot
            while len(self.active_jobs) >= self.max_concurrent_jobs:
                await asyncio.sleep(5)
                await self._cleanup_completed_jobs()
            
            try:
                # Get next job from queue
                priority, job_id, config = await asyncio.wait_for(
                    self.queue.get(), timeout=1.0
                )
                
                # Start training task
                task = asyncio.create_task(
                    self._run_training_job(job_id, config),
                    name=f"training_{job_id}"
                )
                
                self.active_jobs[job_id] = task
                self.logger.info(f"Started job {job_id} (active: {len(self.active_jobs)})")
                
            except asyncio.TimeoutError:
                # No jobs in queue
                await asyncio.sleep(1)
    
    async def _run_training_job(self, job_id: str, config: TrainingConfig):
        """Execute a single training job."""
        
        try:
            # Update status to running
            self._update_job_status(job_id, JobStatus.RUNNING, started_at=datetime.now())
            
            # Execute training
            engine = TrainingEngine(job_id=job_id, gpu_id=config.gpu_id)
            
            def progress_callback(metrics):
                self._save_metrics(job_id, metrics)
                self._update_progress(job_id, metrics.step, metrics.total_bpt)
            
            artifact = await engine.run_training(config, progress_callback)
            
            # Update status to completed
            self._update_job_status(
                job_id, 
                JobStatus.COMPLETED, 
                completed_at=datetime.now(),
                adapter_path=str(artifact.local_path),
                progress_percent=100.0
            )
            
            self.logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            
            # Update status to failed
            self._update_job_status(
                job_id,
                JobStatus.FAILED,
                completed_at=datetime.now(),
                error_message=str(e)
            )
    
    def get_job_status(self, job_id: str) -> Optional[JobMetadata]:
        """Get current job status."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        (job_id, base_model, config_json, status, progress_percent, priority, created_at, 
         started_at, completed_at, error_message, adapter_path) = row
        
        config = TrainingConfig(**json.loads(config_json))
        
        return JobMetadata(
            job_id=job_id,
            base_model=base_model,
            config=config,
            status=JobStatus(status),
            progress_percent=float(progress_percent or 0.0),
            priority=priority,
            created_at=datetime.fromisoformat(created_at),
            started_at=datetime.fromisoformat(started_at) if started_at else None,
            completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
            error_message=error_message,
            adapter_path=adapter_path
        )
    
    def list_jobs(self, status: Optional[JobStatus] = None, limit: int = 50) -> list:
        """List jobs with optional filtering."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status:
            cursor.execute("""
                SELECT job_id, base_model, status, created_at 
                FROM jobs WHERE status = ?
                ORDER BY created_at DESC LIMIT ?
            """, (status.value, limit))
        else:
            cursor.execute("""
                SELECT job_id, base_model, status, created_at 
                FROM jobs ORDER BY created_at DESC LIMIT ?
            """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            job_id, base_model, status, created_at = row
            results.append({
                'job_id': job_id,
                'base_model': base_model,
                'status': status,
                'created_at': created_at
            })
        
        conn.close()
        return results
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        
        if job_id in self.active_jobs:
            task = self.active_jobs[job_id]
            task.cancel()
            self._update_job_status(job_id, JobStatus.CANCELLED)
            
            self.logger.info(f"Cancelled job {job_id}")
            return True
        
        return False
    
    def _update_job_status(self, job_id: str, status: JobStatus, **kwargs):
        """Update job status in database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query
        fields = ["status = ?"]
        values = [status.value]
        
        for key, value in kwargs.items():
            fields.append(f"{key} = ?")
            values.append(value)
        
        values.append(job_id)
        
        query = f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?"
        cursor.execute(query, values)
        conn.commit()
        conn.close()
    
    def _save_metrics(self, job_id: str, metrics: 'TrainingMetrics'):
        """Save training metrics to database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics (job_id, step, data_bpt, param_bpt, s_ratio, 
                               lambda_value, loss)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            job_id, metrics.step, metrics.data_bpt, metrics.param_bpt,
            metrics.s_ratio, metrics.lambda_value, metrics.loss
        ))
        
        conn.commit()
        conn.close()
    
    def _update_progress(self, job_id: str, step: int, total_bpt: float):
        """Update job progress in database."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Derive target steps from stored config (fallback to 1000)
        cursor.execute("SELECT config_json FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()

        target_steps = 1000
        if row:
            try:
                stored_config = json.loads(row[0])
                if stored_config.get("steps"):
                    target_steps = max(1, int(stored_config["steps"]))
            except Exception:
                # Do not block progress updates on malformed config
                target_steps = 1000
        
        progress = min(100.0, (step / target_steps) * 100.0)
        cursor.execute(
            "UPDATE jobs SET progress_percent = ? WHERE job_id = ?",
            (progress, job_id)
        )
        
        conn.commit()
        conn.close()
    
    async def _cleanup_completed_jobs(self):
        """Remove completed jobs from active_jobs dict."""
        
        to_remove = []
        for job_id, task in self.active_jobs.items():
            if task.done():
                to_remove.append(job_id)
                try:
                    await task  # Consume any exceptions
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.error(f"Job {job_id} exception: {e}")
        
        for job_id in to_remove:
            del self.active_jobs[job_id]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} completed jobs")
```

### Step 1.5: FastAPI Entry Point (main.py)

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from pathlib import Path
import uvicorn

from .types import TrainingConfig, JobStatus
from .service.job_queue import JobQueueManager
from .config import APIConfig


app = FastAPI(
    title="SCU Training API",
    description="API for training language models with Shannon Control Unit adaptive regularization",
    version="1.0.0"
)

# Global queue manager (would be better as dependency injection)
queue_manager = None


class JobSubmitRequest(BaseModel):
    """Request body for job submission."""
    base_model: str
    train_data_path: str
    val_data_path: Optional[str] = None
    target_s: float = 0.01
    config_overrides: Optional[dict] = None
    priority: int = 0


class JobSubmitResponse(BaseModel):
    """Response with job ID."""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response with job status."""
    job_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percent: float
    adapter_path: Optional[str] = None
    error_message: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize queue manager on startup."""
    global queue_manager
    
    config = APIConfig()
    queue_manager = JobQueueManager(
        db_path=config.db_path,
        max_concurrent_jobs=config.max_concurrent_jobs
    )
    
    # Start queue processor in background
    asyncio.create_task(queue_manager.start_queue_processor())


@app.post("/jobs/submit", response_model=JobSubmitResponse)
async def submit_job(request: JobSubmitRequest):
    """Submit a new training job."""
    
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Build training configuration
    config = TrainingConfig(
        base_model=request.base_model,
        train_data_path=request.train_data_path,
        val_data_path=request.val_data_path,
        control={"target_s": request.target_s},
        adapter_output_dir=f"./adapters/{request.base_model.split('/')[-1]}-{int(datetime.now().timestamp())}",
        auto_upload_to_hub=False  # Disabled for now
    )
    
    # Apply config overrides
    if request.config_overrides:
        for key, value in request.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    # Submit to queue
    job_id = await queue_manager.submit_job(config, priority=request.priority)
    
    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        message=f"Job {job_id} submitted successfully"
    )


@app.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a training job."""
    
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = queue_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return JobStatusResponse(
        job_id=status.job_id,
        status=status.status.value,
        created_at=status.created_at,
        started_at=status.started_at,
        completed_at=status.completed_at,
        progress_percent=status.progress_percent,
        adapter_path=status.adapter_path,
        error_message=status.error_message
    )


@app.get("/jobs/{job_id}/adapter")
async def download_adapter(job_id: str, background_tasks: BackgroundTasks):
    """Download the trained adapter artifact as a zip file."""
    
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    status = queue_manager.get_job_status(job_id)
    if not status or not status.adapter_path:
        raise HTTPException(status_code=404, detail=f"Adapter for job {job_id} not found")
    
    adapter_path = Path(status.adapter_path)
    if not adapter_path.exists():
        raise HTTPException(status_code=404, detail="Adapter path is missing on disk")
    
    # If directory, stream zipped archive on the fly
    if adapter_path.is_dir():
        import tempfile
        import shutil
        tmp_dir = Path(tempfile.mkdtemp())
        archive_base = tmp_dir / f"{job_id}_adapter"
        archive_file = shutil.make_archive(str(archive_base), "zip", adapter_path)
        archive_path = Path(archive_file)
        filename = f"{job_id}_adapter.zip"
        # Cleanup temp archive after response is sent
        background_tasks.add_task(lambda p=tmp_dir: shutil.rmtree(p, ignore_errors=True))
    else:
        archive_path = adapter_path
        filename = adapter_path.name
    
    return FileResponse(
        path=str(archive_path),
        filename=filename,
        media_type="application/octet-stream"
    )


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = await queue_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Job {job_id} not found or not running")
    
    return {"message": f"Job {job_id} cancelled"}


@app.get("/jobs")
async def list_jobs(status: Optional[JobStatus] = None, limit: int = 50):
    """List training jobs."""
    
    if not queue_manager:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    jobs = queue_manager.list_jobs(status=status, limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/health")
async def health_check():
    """Service health check."""
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(queue_manager.active_jobs) if queue_manager else 0
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## Phase 2: Smart Configuration System

### Step 2.1: Model Size Estimation (service/model_utils.py)

```python
import re
from typing import Dict, Tuple

# Model architecture to parameter count mapping (approximate)
MODEL_PARAM_COUNTS = {
    # Format: "model_name_pattern": (param_count_billion, recommended_config)
    "gpt2": (0.124, {"target_s": 0.005, "lora_r": 8, "batch_size": 8}),
    "distilgpt2": (0.082, {"target_s": 0.005, "lora_r": 8, "batch_size": 16}),
    "microsoft/DialoGPT": (0.124, {"target_s": 0.008, "lora_r": 8, "batch_size": 8}),
    "microsoft/DialoGPT-medium": (0.355, {"target_s": 0.008, "lora_r": 16, "batch_size": 4}),
    "microsoft/DialoGPT-large": (0.774, {"target_s": 0.01, "lora_r": 16, "batch_size": 2}),
    "facebook/blenderbot": (0.4, {"target_s": 0.01, "lora_r": 16, "batch_size": 4}),
    "facebook/blenderbot-3B": (3.0, {"target_s": 0.025, "lora_r": 32, "batch_size": 1}),
    "google/reformer-enwik8": (0.2, {"target_s": 0.008, "lora_r": 16, "batch_size": 4}),
    "EleutherAI/gpt-neo": (0.125, {"target_s": 0.005, "lora_r": 8, "batch_size": 8}),
    "EleutherAI/gpt-neo-1.3B": (1.3, {"target_s": 0.01, "lora_r": 16, "batch_size": 2}),
    "EleutherAI/gpt-neo-2.7B": (2.7, {"target_s": 0.02, "lora_r": 32, "batch_size": 1}),
    "EleutherAI/pythia": (0.7, {"target_s": 0.01, "lora_r": 16, "batch_size": 4}),
    "EleutherAI/pythia-2.8b": (2.8, {"target_s": 0.02, "lora_r": 32, "batch_size": 1}),
    "stanford-crfm/BioMedLM": (2.7, {"target_s": 0.02, "lora_r": 32, "batch_size": 1}),
    # Llama variants
    "Llama-3.2-1": (1.0, {"target_s": 0.01, "lora_r": 16, "batch_size": 4}),
    "Llama-3.2-3": (3.0, {"target_s": 0.02, "lora_r": 32, "batch_size": 2}),
    "Llama-3.1-8": (8.0, {"target_s": 0.03, "lora_r": 64, "batch_size": 1}),
    "Llama-3-70": (70.0, {"target_s": 0.05, "lora_r": 128, "batch_size": 1})
}

def estimate_model_params(model_id: str) -> Tuple[float, Dict]:
    """Estimate parameter count and recommended config for a model."""
    
    model_id_lower = model_id.lower()
    
    # Check exact matches first
    if model_id_lower in MODEL_PARAM_COUNTS:
        return MODEL_PARAM_COUNTS[model_id_lower]
    
    # Check pattern matches (e.g., "Llama-3.2-1" for "meta-llama/Llama-3.2-1B")
    for pattern, (params, config) in MODEL_PARAM_COUNTS.items():
        if pattern.lower() in model_id_lower:
            return params, config.copy()
    
    # Try to extract parameter count from model name
    # Look for patterns like "-1B", "_1b", "1.3b", etc.
    param_match = re.search(r'[-_](\d+(?:\.\d+)?)[bB]', model_id)
    if param_match:
        param_count = float(param_match.group(1))
        return param_count, interpolate_config(param_count)
    
    # Default fallback
    return 1.0, {"target_s": 0.01, "lora_r": 16, "batch_size": 4}

def interpolate_config(param_count_billion: float) -> Dict:
    """Interpolate config for unknown model size."""
    
    if param_count_billion < 0.5:
        return {"target_s": 0.007, "lora_r": 8, "batch_size": 8}
    elif param_count_billion < 1.5:
        return {"target_s": 0.01, "lora_r": 16, "batch_size": 4}
    elif param_count_billion < 4.0:
        return {"target_s": 0.02, "lora_r": 32, "batch_size": 2}
    elif param_count_billion < 10.0:
        return {"target_s": 0.03, "lora_r": 64, "batch_size": 1}
    else:
        return {"target_s": 0.04, "lora_r": 128, "batch_size": 1}
```

### Step 2.2: Configuration Auto-Detection (service/smart_config.py)

```python
from typing import Dict, Optional
from pathlib import Path
from ..types import TrainingConfig, LoRAConfig, ControlConfig

class SmartConfigBuilder:
    """Automatically builds optimal configuration for training."""
    
    def __init__(self):
        from .model_utils import estimate_model_params
        self.estimate_model_params = estimate_model_params
    
    def build_config(
        self,
        base_model: str,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        **user_overrides
    ) -> TrainingConfig:
        """
        Auto-build configuration with smart defaults.
        
        Steps:
        1. Estimate model size and get base config
        2. Analyze dataset size and adjust batch_size
        3. Apply user overrides
        4. Validate configuration
        """
        
        # Step 1: Get base config from model size
        param_count, base_config = self.estimate_model_params(base_model)
        
        # Build configuration with smart defaults
        config_dict = {
            # Model
            "base_model": base_model,
            "train_data_path": train_data_path,
            "val_data_path": val_data_path,
            
            # Control (use model-specific target_s)
            "control": {
                "target_s": base_config["target_s"],
                "Kp": 0.8,
                "Ki": 0.15,
                "lambda_init": 1.0
            },
            
            # LoRA (use model-specific rank)
            "lora": {
                "r": base_config["lora_r"],
                "alpha": base_config["lora_r"] * 2,  # alpha = r * 2 is standard
                "dropout": 0.05
            },
            
            # Training hyperparams (model-specific)
            "batch_size": base_config["batch_size"],
            "learning_rate": self._estimate_lr(param_count, base_config["batch_size"]),
            "gradient_accumulation_steps": self._estimate_grad_accum(base_config["batch_size"]),
            "block_size": self._estimate_block_size(param_count),
            "steps": None,  # Use epochs
            "epochs": 1,  # Default to 1 epoch
            "prior_sigma": 0.01,
            
            # Output
            "adapter_output_dir": f"./adapters/{base_model.split('/')[-1]}-{int(Path.cwd().name) or 0}",
            "auto_upload_to_hub": False,
            
            # Hardware
            "use_4bit": True,
            "use_fp16": torch.cuda.is_available(),
            "gpu_id": None
        }
        
        # Step 2: Adjust for dataset size
        dataset_size = self._estimate_dataset_size(train_data_path)
        if dataset_size < 1_000_000:  # Small dataset
            config_dict["batch_size"] = min(config_dict["batch_size"], 4)
            config_dict["control"]["target_s"] *= 0.7  # More conservative
        elif dataset_size > 10_000_000:  # Large dataset
            config_dict["epochs"] = 1  # No need for multiple epochs
        
        # Step 3: Apply user overrides
        for key, value in user_overrides.items():
            if key in config_dict:
                if isinstance(config_dict[key], dict) and isinstance(value, dict):
                    config_dict[key].update(value)
                else:
                    config_dict[key] = value
        
        # Create config object
        return TrainingConfig(**config_dict)
    
    def _estimate_lr(self, param_count: float, batch_size: int) -> float:
        """Estimate optimal learning rate based on model size and batch."""
        
        # Simple heuristic: smaller models → higher LR, larger batches → higher LR
        base_lr = 2e-5 if param_count > 5.0 else 5e-5
        batch_factor = (batch_size / 4.0) ** 0.5  # Scale LR with sqrt(batch size)
        
        return base_lr * batch_factor
    
    def _estimate_grad_accum(self, batch_size: int) -> int:
        """Estimate gradient accumulation steps to maintain effective batch."""
        
        target_effective_batch = 16  # Good default for most cases
        
        if batch_size >= target_effective_batch:
            return 1
        else:
            return target_effective_batch // batch_size
    
    def _estimate_block_size(self, param_count: float) -> int:
        """Estimate optimal block size (sequence length)."""
        
        # Larger models can handle longer sequences without OOM
        if param_count < 2.0:
            return 1024
        elif param_count < 7.0:
            return 2048
        else:
            return 4096
    
    def _estimate_dataset_size(self, data_path: str) -> int:
        """Estimate dataset size in tokens."""
        
        from pathlib import Path
        
        path = Path(data_path)
        if not path.exists():
            return 1_000_000  # Default assumption
        
        # Rough estimate: 1 token ≈ 0.75 bytes
        size_bytes = path.stat().st_size
        estimated_tokens = int(size_bytes * 0.75)
        
        return estimated_tokens
```

### Step 2.3: Configuration Validation (service/validator.py)

```python
from typing import List, Dict
from pydantic import ValidationError
import torch
import psutil
import json
from ..types import TrainingConfig

class ValidationIssue:
    """Represents a validation warning or error."""
    
    def __init__(self, message: str, is_error: bool = False, suggestion: str = ""):
        self.message = message
        self.is_error = is_error
        self.suggestion = suggestion

class ValidationReport:
    """Complete validation report."""
    
    def __init__(self, issues: List[ValidationIssue]):
        self.issues = issues
        self.passed = len([i for i in issues if i.is_error]) == 0
    
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.is_error]
    
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if not i.is_error]

class ConfigValidator:
    """Validates training configurations before job submission."""
    
    def __init__(self):
        self.allowed_models = [
            "gpt2", "microsoft/DialoGPT", "facebook/blenderbot",
            "EleutherAI/gpt-neo", "EleutherAI/pythia",
            "Llama-3", "Llama-3.1", "Llama-3.2",
            "gemma", "stanford-crfm/BioMedLM"
        ]
    
    def validate(self, config: TrainingConfig) -> ValidationReport:
        """Run all validation checks."""
        
        issues = []
        
        # Run all check methods
        issues.extend(self._check_model_whitelist(config))
        issues.extend(self._check_dataset_accessible(config))
        issues.extend(self._check_gpu_availability(config))
        issues.extend(self._check_memory_footprint(config))
        issues.extend(self._check_hyperparameters(config))
        issues.extend(self._check_output_directory(config))
        issues.extend(self._check_lora_configuration(config))
        
        return ValidationReport(issues)
    
    def _check_model_whitelist(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check if model is in allowed list (security + compatibility)."""
        
        model_id = config.base_model.lower()
        
        is_allowed = any(
            allowed.lower() in model_id 
            for allowed in self.allowed_models
        )
        
        if not is_allowed:
            return [ValidationIssue(
                message=f"Model {config.base_model} not in allowed list. "
                       f"This may be untested or incompatible with SCU.",
                is_error=False,
                suggestion="Use a supported model: " + ", ".join(self.allowed_models[:5]) + "..."
            )]
        
        return []
    
    def _check_dataset_accessible(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Verify training data is accessible."""
        
        from pathlib import Path
        
        data_path = Path(config.train_data_path)
        
        # Check if it's a URL
        if config.train_data_path.startswith("http://") or config.train_data_path.startswith("https://"):
            import urllib.request
            try:
                urllib.request.urlopen(config.train_data_path, timeout=5)
                return []
            except Exception as e:
                return [ValidationIssue(
                    message=f"Cannot access dataset URL: {e}",
                    is_error=True,
                    suggestion="Check URL and network connectivity"
                )]
        
        # Check local file
        if not data_path.exists():
            return [ValidationIssue(
                message=f"Dataset file not found: {config.train_data_path}",
                is_error=True,
                suggestion="Provide absolute path or working directory-correct relative path"
            )]
        
        if data_path.stat().st_size == 0:
            return [ValidationIssue(
                message="Dataset file is empty",
                is_error=True,
                suggestion="Check dataset generation"
            )]
        
        return []
    
    def _check_gpu_availability(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check GPU availability and compatibility."""
        
        issues = []
        
        # Check if CUDA requested but not available
        if config.use_4bit and not torch.cuda.is_available():
            issues.append(ValidationIssue(
                message="4-bit quantization requested but CUDA not available",
                is_error=False,
                suggestion="set use_4bit=False or run on GPU-enabled machine"
            ))
        
        if config.use_fp16 and not torch.cuda.is_available():
            issues.append(ValidationIssue(
                message="FP16 requested but CUDA not available",
                is_error=False,
                suggestion="set use_fp16=False or run on GPU-enabled machine"
            ))
        
        # Check if specific GPU requested
        if config.gpu_id is not None:
            if not torch.cuda.is_available():
                issues.append(ValidationIssue(
                    message=f"GPU {config.gpu_id} requested but no CUDA GPUs available",
                    is_error=True,
                    suggestion="Run on machine with GPU or set gpu_id=None"
                ))
            elif config.gpu_id >= torch.cuda.device_count():
                issues.append(ValidationIssue(
                    message=f"GPU {config.gpu_id} requested but only {torch.cuda.device_count()} GPUs available",
                    is_error=True,
                    suggestion=f"Choose GPU between 0 and {torch.cuda.device_count() - 1}"
                ))
        
        return issues
    
    def _check_memory_footprint(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Estimate GPU memory requirements."""
        
        if not torch.cuda.is_available():
            return []  # Can't check without CUDA
        
        issues = []
        
        # Estimate model size
        from .model_utils import estimate_model_params
        param_count, base_config = estimate_model_params(config.base_model)
        
        # Memory calculation (rough estimation)
        # Model weights: param_count * 4 bytes (fp32) or 2 bytes (fp16)
        dtype_bytes = 2 if config.use_fp16 else 4
        model_memory_gb = (param_count * 1e9 * dtype_bytes) / (1024**3)
        
        # LoRA overhead
        lora_overhead_gb = (config.lora.r * param_count * 0.001) / 1024 + 0.5
        
        # Batch memory (activations) - roughly proportional to model size
        batch_memory_gb = model_memory_gb * (config.batch_size / 4.0) * 0.3
        
        # Total estimated memory
        total_memory_gb = model_memory_gb + lora_overhead_gb + batch_memory_gb
        
        # Get available GPU memory
        allocated = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        reserved = torch.cuda.memory_reserved(0) if torch.cuda.is_available() else 0
        available_gb = (torch.cuda.get_device_properties(0).total_memory - reserved) / (1024**3)
        
        if total_memory_gb > available_gb:
            issues.append(ValidationIssue(
                message=f"Estimated {total_memory_gb:.1f}GB GPU memory needed, "
                       f"but only {available_gb:.1f}GB available",
                is_error=True,
                suggestion="Reduce batch_size, lora.r, or block_size; or use a larger GPU"
            ))
        elif total_memory_gb > available_gb * 0.9:
            issues.append(ValidationIssue(
                message=f"High memory usage: {total_memory_gb:.1f}GB / {available_gb:.1f}GB",
                is_error=False,
                suggestion="Consider reducing batch_size for safety margin"
            ))
        
        # RAM check for data loading
        system_memory = psutil.virtual_memory()
        if system_memory.available < 2 * 1024**3:  # < 2GB RAM free
            issues.append(ValidationIssue(
                message=f"Low system RAM: {system_memory.available / 1024**3:.1f}GB available",
                is_error=False,
                suggestion="Free up RAM or use a machine with more memory for data loading"
            ))
        
        return issues
    
    def _check_hyperparameters(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check hyperparameter ranges and compatibility."""
        
        issues = []
        
        # Learning rate sanity check
        if config.learning_rate > 1e-3:
            issues.append(ValidationIssue(
                message=f"Learning rate {config.learning_rate} seems high",
                is_error=False,
                suggestion="Typical range is 1e-7 to 1e-4 for LoRA fine-tuning"
            ))
        
        # Check target_s is reasonable
        if config.control.target_s > 0.1:
            issues.append(ValidationIssue(
                message=f"Target S {config.control.target_s} is very high (>10%)",
                is_error=False,
                suggestion="Typical range is 0.005-0.05 (0.5% - 5%)"
            ))
        elif config.control.target_s < 0.001:
            issues.append(ValidationIssue(
                message=f"Target S {config.control.target_s} is very low (<0.1%)",
                is_error=False,
                suggestion="This may lead to over-regularization; typical minimum is 0.005"
            ))
        
        # Check steps/epochs
        if config.steps is None and config.epochs is None:
            issues.append(ValidationIssue(
                message="Neither steps nor epochs specified",
                is_error=True,
                suggestion="Set either steps or epochs"
            ))
        
        # Batch size vs accumulation
        effective_batch = config.batch_size * config.gradient_accumulation_steps
        if effective_batch < 4:
            issues.append(ValidationIssue(
                message=f"Effective batch size {effective_batch} is small",
                is_error=False,
                suggestion="Increase batch_size or gradient_accumulation_steps for better stability"
            ))
        
        return issues
    
    def _check_output_directory(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check output directory is writable."""
        
        from pathlib import Path
        
        output_dir = Path(config.adapter_output_dir)
        
        try:
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            # Try to create temp file to test write access
            test_file = output_dir.parent / ".scu_write_test"
            test_file.touch()
            test_file.unlink()
            return []
        except Exception as e:
            return [ValidationIssue(
                message=f"Cannot write to output directory: {e}",
                is_error=True,
                suggestion="Check permissions or choose a different directory"
            )]
    
    def _check_lora_configuration(self, config: TrainingConfig) -> List[ValidationIssue]:
        """Check LoRA configuration is reasonable."""
        
        issues = []
        
        # LoRA rank vs model size
        if config.lora.r > 64 and param_count < 2.0:
            issues.append(ValidationIssue(
                message=f"LoRA rank {config.lora.r} is high for small model",
                is_error=False,
                suggestion=f"Consider reducing to 16-32 for models < 2B parameters"
            ))
        
        # Alpha vs rank ratio
        alpha_to_rank = config.lora.alpha / config.lora.r
        if alpha_to_rank < 1.5 or alpha_to_rank > 3.0:
            issues.append(ValidationIssue(
                message=f"LoRA alpha/rank ratio {alpha_to_rank:.1f} is unusual",
                is_error=False,
                suggestion="Typical ratio is 2.0 (alpha = 2 * rank)"
            ))
        
        return issues
```

---

## Phase 3: Client Implementation

### Step 3.1: Python SDK (client/sync_client.py)

```python
import requests
import time
from typing import Optional, Dict, Any, Iterator
from pathlib import Path
import json

from ..types import JobStatusResponse, JobSubmitResponse

class SCUClient:
    """
    Synchronous client for SCU Training API.
    Handles job submission, monitoring, and result retrieval.
    """
    
    def __init__(self, api_url: str = "http://localhost:8000", api_key: Optional[str] = None):
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def submit_training_job(
        self,
        base_model: str,
        train_data: str,
        val_data: Optional[str] = None,
        target_s: float = 0.01,
        config_overrides: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        wait_for_completion: bool = False,
        poll_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Submit a training job.
        
        Args:
            base_model: HuggingFace model identifier
            train_data: Path or URL to training data
            val_data: Path to validation data (optional)
            target_s: Target information ratio
            config_overrides: Additional configuration overrides
            priority: Job priority (higher = sooner)
            wait_for_completion: Block until job completes
            poll_interval: Seconds between status checks when waiting
            
        Returns:
            Job information including ID
        """
        
        url = f"{self.api_url}/jobs/submit"
        
        payload = {
            "base_model": base_model,
            "train_data_path": str(train_data),
            "val_data_path": str(val_data) if val_data else None,
            "target_s": target_s,
            "config_overrides": config_overrides or {},
            "priority": priority
        }
        
        response = self.session.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if wait_for_completion:
            job_id = result["job_id"]
            print(f"Job submitted: {job_id}. Waiting for completion...")
            
            while True:
                status = self.get_job_status(job_id)
                
                if status["status"] == "completed":
                    print(f"Job {job_id} completed!")
                    break
                elif status["status"] == "failed":
                    raise RuntimeError(f"Job failed: {status.get('error_message')}")
                elif status["status"] == "cancelled":
                    raise RuntimeError("Job was cancelled")
                
                # Progress indicator
                progress = status.get("progress_percent", 0)
                print(f"Progress: {progress:.1f}%")
                
                time.sleep(poll_interval)
        
        return result
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get current status of a job."""
        
        url = f"{self.api_url}/jobs/{job_id}/status"
        response = self.session.get(url)
        response.raise_for_status()
        
        return response.json()
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> list:
        """List jobs with optional status filtering."""
        
        url = f"{self.api_url}/jobs"
        params = {"limit": limit}
        
        if status:
            params["status"] = status
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        return response.json()["jobs"]
    
    def cancel_job(self, job_id: str):
        """Cancel a running job."""
        
        url = f"{self.api_url}/jobs/{job_id}/cancel"
        response = self.session.post(url)
        response.raise_for_status()
        
        return response.json()
    
    def get_adapter(self, job_id: str, output_dir: Path):
        """Download trained adapter to local directory."""
        
        url = f"{self.api_url}/jobs/{job_id}/adapter"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        response = self.session.get(url, stream=True)
        if response.status_code == 404:
            raise ValueError(f"No adapter available for job {job_id}")
        response.raise_for_status()
        
        # Derive filename from content-disposition or fall back to job id
        filename = None
        content_disp = response.headers.get("content-disposition", "")
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip().strip('"')
        if not filename:
            filename = f"{job_id}_adapter.zip"
        
        adapter_file = output_dir / filename
        with open(adapter_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        return adapter_file
    
    def health_check(self) -> bool:
        """Check if API service is healthy."""
        
        try:
            url = f"{self.api_url}/health"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return True
        except Exception:
            return False
    
    def get_training_metrics(self, job_id: str) -> Iterator[Dict[str, Any]]:
        """Stream training metrics over time."""
        
        # For now, read from CSV file
        # In production, would poll API endpoint
        status = self.get_job_status(job_id)
        adapter_path = status.get("adapter_path")
        
        if not adapter_path:
            return
        
        metrics_path = Path(adapter_path) / "training_metrics.csv"
        
        if not metrics_path.exists():
            return
        
        import csv
        with open(metrics_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


class SCUClientError(Exception):
    """Custom exception for SCU client errors."""
    pass
```

### Step 3.2: CLI Interface (cli/main.py)

```python
import click
import json
import sys
from pathlib import Path

from ..client.sync_client import SCUClient


@click.group()
@click.option("--api-url", default="http://localhost:8000", help="SCU API URL")
@click.option("--api-key", help="API key for authentication")
@click.pass_context
def cli(ctx, api_url, api_key):
    """SCU Training CLI - Train models with adaptive regularization."""
    ctx.ensure_object(dict)
    ctx.obj["client"] = SCUClient(api_url=api_url, api_key=api_key)


@cli.command()
@click.option("--base-model", required=True, help="HuggingFace model ID")
@click.option("--train-data", required=True, type=click.Path(exists=True), help="Training data path")
@click.option("--val-data", type=click.Path(exists=True), help="Validation data path")
@click.option("--target-s", default=0.01, type=float, help="Target information ratio")
@click.option("--name", help="Job name for tracking")
@click.option("--priority", default=0, type=int, help="Job priority (higher = sooner)")
@click.option("--config", type=click.Path(exists=True), help="JSON config file with overrides")
@click.option("--wait", is_flag=True, help="Wait for job completion")
@click.pass_context
def train(ctx, base_model, train_data, val_data, target_s, name, priority, config, wait):
    """Submit a training job."""
    
    client = ctx.obj["client"]
    
    # Check API health
    if not client.health_check():
        click.echo("Error: SCU API not available", err=True)
        sys.exit(1)
    
    # Load config overrides
    config_overrides = {}
    if config:
        with open(config) as f:
            config_overrides = json.load(f)
    
    click.echo(f"Submitting training job for {base_model}...")
    
    try:
        result = client.submit_training_job(
            base_model=base_model,
            train_data=train_data,
            val_data=val_data,
            target_s=target_s,
            config_overrides=config_overrides,
            priority=priority,
            wait_for_completion=wait,
            poll_interval=10
        )
        
        click.echo(f"Job submitted: {result['job_id']}")
        
        if name:
            click.echo(f"Job name: {name}")
        
        click.echo(f"Status: {result['status']}")
        click.echo(f"Track progress with: scu status {result['job_id']}")
        
    except Exception as e:
        click.echo(f"Error submitting job: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("job-id")
@click.pass_context
def status(ctx, job_id):
    """Get job status."""
    
    client = ctx.obj["client"]
    
    try:
        job_status = client.get_job_status(job_id)
        
        click.echo(f"Job: {job_id}")
        click.echo(f"Status: {job_status['status']}")
        click.echo(f"Created: {job_status['created_at']}")
        
        if job_status.get('started_at'):
            click.echo(f"Started: {job_status['started_at']}")
        
        if job_status.get('completed_at'):
            click.echo(f"Completed: {job_status['completed_at']}")
        
        if job_status.get('error_message'):
            click.echo(f"Error: {job_status['error_message']}")
        
        if job_status.get('adapter_path'):
            click.echo(f"Adapter: {job_status['adapter_path']}")
        
    except Exception as e:
        click.echo(f"Error getting status: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--status", type=click.Choice(['queued', 'running', 'completed', 'failed', 'cancelled']), help="Filter by status")
@click.option("--limit", default=20, help="Maximum number of jobs to show")
@click.pass_context
def jobs(ctx, status, limit):
    """List all jobs."""
    
    client = ctx.obj["client"]
    
    try:
        jobs = client.list_jobs(status=status, limit=limit)
        
        if not jobs:
            click.echo("No jobs found")
            return
        
        # Print table
        click.echo(f"{'ID':<10} {'Model':<30} {'Status':<12} {'Created':<20}")
        click.echo("-" * 80)
        
        for job in jobs:
            model_name = job['base_model'].split('/')[-1][:28]
            created = job['created_at'][:19] if len(job['created_at']) > 19 else job['created_at']
            click.echo(f"{job['job_id']:<10} {model_name:<30} {job['status']:<12} {created:<20}")
        
    except Exception as e:
        click.echo(f"Error listing jobs: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("job-id")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory for adapter")
@click.pass_context
def download(ctx, job_id, output_dir):
    """Download trained adapter."""
    
    client = ctx.obj["client"]
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        adapter_path = client.get_adapter(job_id, output_path)
        click.echo(f"Adapter downloaded to: {adapter_path}")
        
    except Exception as e:
        click.echo(f"Error downloading adapter: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--base-model", required=True, help="HuggingFace model ID")
def recommend_config(base_model):
    """Show recommended configuration for a model."""
    
    from ..service.smart_config import SmartConfigBuilder
    
    builder = SmartConfigBuilder()
    config = builder.build_config(base_model, "dummy_data.txt")
    
    click.echo(f"Recommended configuration for {base_model}:")
    click.echo()
    click.echo(f"Target S: {config.control.target_s}")
    click.echo(f"LoRA rank: {config.lora.r}")
    click.echo(f"Batch size: {config.batch_size}")
    click.echo(f"Learning rate: {config.learning_rate}")
    click.echo(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    click.echo(f"Block size: {config.block_size}")
    click.echo(f"4-bit quantization: {config.use_4bit}")


if __name__ == "__main__":
    cli()
```

---

## Testing Strategy

### Unit Tests (tests/test_training_engine.py)

```python
import pytest
import tempfile
import torch
from pathlib import Path

from scu_api.types import TrainingConfig
from scu_api.service.training_engine import TrainingEngine


@pytest.fixture
def sample_config():
    """Create a minimal test configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test training data. " * 100)  # 2000 tokens approx
        temp_path = f.name
    
    return TrainingConfig(
        base_model="gpt2",  # Small model for testing
        train_data_path=temp_path,
        val_data_path=None,
        adapter_output_dir="./test_adapters",
        control={"target_s": 0.01},
        lora={"r": 4, "alpha": 8, "dropout": 0.05},
        batch_size=2,
        steps=10,  # Very short training for test
        epochs=None,
        use_4bit=False,  # Disable quantization for CPU testing
        use_fp16=False,
        gpu_id=None
    )


@pytest.mark.asyncio
async def test_training_engine_initialization():
    """Test that TrainingEngine can be initialized."""
    engine = TrainingEngine(job_id="test_job", gpu_id=None)
    assert engine.job_id == "test_job"
    assert engine.accelerator is None


@pytest.mark.asyncio
async def test_training_loop_complete(sample_config):
    """Test complete training loop executes without errors."""
    
    # Skip if no GPU and model is too large
    if not torch.cuda.is_available():
        sample_config.steps = 2  # Very short
        sample_config.base_model = "sshleifer/tiny-gpt2"  # Extra small
    
    engine = TrainingEngine(job_id="test_training", gpu_id=None)
    
    metrics_received = []
    
    def progress_callback(metrics):
        metrics_received.append(metrics)
    
    # Run training
    artifact = await engine.run_training(sample_config, progress_callback)
    
    # Verify results
    assert artifact.job_id == "test_training"
    assert artifact.base_model == sample_config.base_model
    assert artifact.local_path.exists()
    assert (artifact.local_path / "adapter_config.json").exists()
    
    # Check metrics were reported
    assert len(metrics_received) > 0
    assert all(hasattr(m, 's_ratio') for m in metrics_received)
    
    # Cleanup
    import shutil
    shutil.rmtree(sample_config.adapter_output_dir, ignore_errors=True)


def test_device_detection():
    """Test automatic device detection."""
    from scu_api.service.training_engine import _get_device_map, _get_torch_dtype
    
    config = TrainingConfig(
        base_model="gpt2",
        train_data_path="dummy.txt",
        adapter_output_dir="./dummy",
        use_4bit=False,
        use_fp16=False
    )
    
    device_map = _get_device_map(config)
    torch_dtype = _get_torch_dtype(config)
    
    # Should not raise errors
    assert device_map is not None
    assert torch_dtype in [torch.float32, torch.float16]
```

### Integration Tests (tests/test_api_integration.py)

```python
import pytest
import subprocess
import time
import requests
import tempfile
import json

@pytest.fixture(scope="module")
def api_server():
    """Start API server for testing."""
    
    # Create temp DB
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Start server in background
    env = {"SCU_DB_PATH": db_path, "SCU_MAX_CONCURRENT_JOBS": "1"}
    proc = subprocess.Popen([
        "python", "-m", "scu_api.main"
    ], env={**env, **os.environ}, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for startup
    for _ in range(30):  # 30 seconds timeout
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        proc.kill()
        raise RuntimeError("API server failed to start")
    
    yield proc
    
    # Cleanup
    proc.terminate()
    proc.wait()
    os.unlink(db_path)


def test_submit_job(api_server):
    """Test job submission via API."""
    
    # Create test data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("test data " * 100)
        data_path = f.name
    
    payload = {
        "base_model": "sshleifer/tiny-gpt2",
        "train_data_path": data_path,
        "target_s": 0.01,
        "config_overrides": {
            "steps": 2,
            "lora": {"r": 2, "alpha": 4}
        }
    }
    
    response = requests.post("http://localhost:8000/jobs/submit", json=payload)
    
    assert response.status_code == 200
    result = response.json()
    assert "job_id" in result
    assert result["status"] == "queued"
    
    # Cleanup data file
    os.unlink(data_path)
```

---

## Deployment & Operations

### Docker Configuration

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY scu_api/ ./scu_api/

# Create directories for adapters and logs
RUN mkdir -p /data/adapters /data/logs /data/checkpoints

# Expose API port
EXPOSE 8000

# Environment variables
ENV PYTHONPATH=/app
ENV SCU_ADAPTERS_DIR=/data/adapters
ENV SCU_LOGS_DIR=/data/logs
ENV SCU_CHECKPOINTS_DIR=/data/checkpoints

# Start API server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "scu_api.main:app"]
```

### Docker Compose (production)

```yaml
version: '3.8'

services:
  scu_api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SCU_DB_URL=postgresql://scu:password@postgres:5432/scu_db
      - SCU_HF_TOKEN=${HF_TOKEN}
      - SCU_MAX_CONCURRENT_JOBS=2
      - CUDA_VISIBLE_DEVICES=0,1
    volumes:
      - ./adapters:/data/adapters
      - ./logs:/data/logs
      - ~/.cache/huggingface:/root/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    depends_on:
      - postgres
    networks:
      - scu_network

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: scu
      POSTGRES_PASSWORD: password
      POSTGRES_DB: scu_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - scu_network

  redis:
    image: redis:7-alpine
    networks:
      - scu_network

volumes:
  postgres_data:

networks:
  scu_network:
    driver: bridge
```

---

## Performance Optimization Checklist

- [ ] Use `torch.compile()` on model (PyTorch 2.0+) for 20-30% speedup
- [ ] Implement gradient accumulation to support larger effective batch sizes
- [ ] Add dataset caching to avoid re-tokenizing same data
- [ ] Use `accelerator.save_state()` for efficient checkpointing
- [ ] Profile memory usage and optimize buffer sizes
- [ ] Add metrics aggregation to reduce DB writes
- [ ] Implement HTTP connection pooling in client
- [ ] Use `aiofiles` for async file I/O in API
- [ ] Add Redis caching for frequently accessed job statuses
- [ ] Implement GPU memory defragmentation between jobs

---

## Success Metrics & Validation

**Before Launch:**
- [ ] Training on 3+ different model architectures completes successfully
- [ ] BPT improvement within 10% of research paper baselines
- [ ] API handles 10+ concurrent job submissions without errors
- [ ] Configuration validation catches 90%+ of common mistakes
- [ ] Client SDK can submit job, monitor, and retrieve adapter
- [ ] Documentation has 3+ complete examples
- [ ] Integration tests pass in CI/CD

**After Launch (30 days):**
- [ ] 50+ users have successfully trained models
- [ ] Average time from submission to first training step < 30 seconds
- [ ] Job success rate > 85% (excluding user configuration errors)
- [ ] User feedback: "Easier than manual HF fine-tuning"

---

## Handover Summary

**Code is ready for:**
1. Direct implementation by ML engineering team
2. Integration with existing infrastructure
3. Phased rollout starting with Phase 1 (CLI/local API)
4. Production deployment with Docker
5. Scaling to multiple GPUs and concurrent users

**Key Implementation Priority:**
1. Start with `TrainingEngine` refactor (most critical)
2. Add `JobQueueManager` for async execution
3. Build FastAPI endpoints for remote access
4. Create Python SDK for easy integration
5. Add smart configuration and validation
6. Deploy with monitoring and scaling

**Start Here:** `scu_api/service/training_engine.py` - Refactor existing `train_scu.py` into this class structure.
