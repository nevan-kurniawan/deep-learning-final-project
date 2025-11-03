@echo off
ECHO Starting evaluation for all 8 experimental runs...

SET "PYTHON_EXE=.\.venv\Scripts\python.exe" REM Adjust if your venv path is different
SET "EVAL_MODULE=src.evaluate"
SET "CONFIG_DIR=.\configs"
SET "MODEL_OUTPUT_DIR=.\outputs\models"

:: Capture all arguments passed to this batch script (e.g., --run_diagnostics)
SET "EXTRA_EVAL_ARGS=%*"

:: Define all 8 experiment config base names (without .yaml)
SET "EXPERIMENTS=deit_full_finetune deit_linear_probe deit_static_lora deit_moe_lora vit_full_finetune vit_linear_probe vit_static_lora vit_moe_lora"

:: --- Execution Loop ---
FOR %%G IN (%EXPERIMENTS%) DO (
    ECHO.
    ECHO ==================================================
    ECHO Starting Evaluation for: %%G
    ECHO ==================================================

    :: Construct the path directly in the command
    %PYTHON_EXE% -m %EVAL_MODULE% --model_path "%MODEL_OUTPUT_DIR%\%%G\best_%%G_model.pth" %EXTRA_EVAL_ARGS%

    IF ERRORLEVEL 1 (
        ECHO Evaluation failed for %%G. Continuing with next experiment...
    )
)

ECHO.
ECHO ==================================================
ECHO All evaluations complete.
ECHO ==================================================

:EOF
ECHO Script finished.