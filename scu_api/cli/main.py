import click

from scu_api.client.sync_client import SCUClient


@click.group()
@click.option("--api-url", default="http://localhost:8000", help="SCU API base URL")
@click.pass_context
def cli(ctx, api_url):
    ctx.obj = {"client": SCUClient(api_url)}


@cli.command()
@click.option("--base-model", required=True, help="HuggingFace model id")
@click.option("--train-data", required=True, type=click.Path(exists=True), help="Training data path")
@click.option("--steps", type=int, default=None, help="Number of training steps")
@click.option("--epochs", type=int, default=1, help="Epochs (if steps not set)")
@click.option("--batch-size", type=int, default=1)
@click.option("--lr", type=float, default=5e-5)
@click.option("--target-s", type=float, default=0.01)
@click.option("--kp", type=float, default=0.8)
@click.option("--ki", type=float, default=0.15)
@click.option("--lora-r", type=int, default=16)
@click.option("--lora-alpha", type=int, default=32)
@click.option("--lora-dropout", type=float, default=0.05)
@click.option("--fp16/--no-fp16", default=True)
@click.option("--use-unsloth", is_flag=True, default=False, help="Load via Unsloth fast path")
@click.option("--wait", is_flag=True, help="Wait for completion")
@click.option("--adapter-out", default=None, help="Optional adapter output path")
@click.option("--gradient-accumulation-steps", type=int, default=4)
@click.option("--block-size", type=int, default=1024)
@click.option("--seed", type=int, default=42)
@click.option("--max-texts", type=int, default=None)
@click.option("--auto-config", is_flag=True, default=False, help="Fetch suggested settings before submit")
@click.pass_context
def train(
    ctx,
    base_model,
    train_data,
    steps,
    epochs,
    batch_size,
    lr,
    target_s,
    kp,
    ki,
    lora_r,
    lora_alpha,
    lora_dropout,
    fp16,
    use_unsloth,
    wait,
    adapter_out,
    gradient_accumulation_steps,
    block_size,
    seed,
    max_texts,
    auto_config,
):
    """Submit a training job."""

    client = ctx.obj["client"]
    payload = {
        "base_model": base_model,
        "train_data": str(train_data),
        "steps": steps,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "target_s": target_s,
        "kp": kp,
        "ki": ki,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "fp16": fp16,
        "use_unsloth": use_unsloth,
        "adapter_out": adapter_out,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "block_size": block_size,
        "seed": seed,
        "max_texts": max_texts,
    }

    if auto_config:
        auto = client.auto_configure(base_model, str(train_data))
        suggested = auto.get("suggested_config", {})
        # Merge suggested defaults - they override CLI defaults but not explicit values
        merged = {**payload}
        merged.update(suggested)
        merged["train_data"] = str(train_data)
        merged["base_model"] = base_model
        payload = merged

    # Remove None values so server defaults apply cleanly
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        job = client.submit_job(**payload, wait=wait)
        click.echo(f"Job {job['job_id']} submitted (status={job.get('status', 'queued')})")
        if wait and job.get("status") == "succeeded":
            click.echo(f"Adapter saved to: {job.get('adapter_path')}")
    except Exception as e:  # pragma: no cover - CLI surface
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.argument("job_id")
@click.pass_context
def status(ctx, job_id):
    """Show status for a job."""
    client = ctx.obj["client"]
    try:
        info = client.get_job_status(job_id)
        click.echo(info)
    except Exception as e:  # pragma: no cover - CLI surface
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.pass_context
def jobs(ctx):
    """List recent jobs."""
    client = ctx.obj["client"]
    try:
        for job in client.list_jobs():
            click.echo(f"{job['job_id']}: {job.get('status', 'unknown')} ({job.get('progress', 0)}%)")
    except Exception as e:  # pragma: no cover - CLI surface
        click.echo(f"Error listing jobs: {e}", err=True)


@cli.command()
@click.argument("job_id")
@click.option("--output", default="adapters", type=click.Path(), help="File or directory to save adapter")
@click.pass_context
def download(ctx, job_id, output):
    """Download adapter artifact for a job."""
    client = ctx.obj["client"]
    try:
        path = client.download_adapter(job_id, output)
        click.echo(f"Saved to {path}")
    except Exception as e:  # pragma: no cover - CLI surface
        click.echo(f"Error: {e}", err=True)


@cli.command(name="auto-config")
@click.option("--model-id", required=True)
@click.option("--train-data", type=click.Path())
@click.pass_context
def auto_config(ctx, model_id, train_data):
    """Generate suggested training config for a model."""
    client = ctx.obj["client"]
    try:
        result = client.auto_configure(model_id, str(train_data) if train_data else None)
        click.echo(result)
    except Exception as e:  # pragma: no cover - CLI surface
        click.echo(f"Error: {e}", err=True)


@cli.command()
@click.pass_context
def health(ctx):
    """Check API health."""
    client = ctx.obj["client"]
    try:
        click.echo(client.health())
    except Exception as e:  # pragma: no cover - CLI surface
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    cli()
