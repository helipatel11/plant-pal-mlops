# src/checks/simple_checks.py

def run_basic_checks(train_gen, val_gen):
    """Basic sanity checks for data generators."""
    
    # Check number of classes
    assert train_gen.num_classes == 2, "Train generator must have 2 classes"
    assert val_gen.num_classes == 2, "Validation generator must have 2 classes"
    
    # Check generator outputs
    x_batch, y_batch = next(train_gen)
    assert x_batch.shape[1:] == (224,224,3), f"Unexpected image shape: {x_batch.shape[1:]}"
    assert len(y_batch.shape) == 1, f"Expected labels to be 1D, got {y_batch.shape}"
    
    print(f"✅ Data checks passed: {train_gen.samples} train samples, {val_gen.samples} val samples")
