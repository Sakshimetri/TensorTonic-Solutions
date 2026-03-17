def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """

    # After training ends → constant final_lr
    if step > total_steps:
        return float(final_lr)

    # Warmup phase
    if warmup_steps > 0 and step < warmup_steps:
        lr = (step / warmup_steps) * initial_lr
        return float(lr)

    # Decay phase
    if total_steps == warmup_steps:
        return float(final_lr)  # avoid division by zero

    lr = final_lr + (initial_lr - final_lr) * (
        (total_steps - step) / (total_steps - warmup_steps)
    )

    return float(lr)