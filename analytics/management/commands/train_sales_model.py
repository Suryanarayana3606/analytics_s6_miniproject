from django.core.management.base import BaseCommand
from analytics.ml.train_model import train_and_save_model
from analytics.ml.predict import run_predictions


class Command(BaseCommand):
    help = 'Train ML forecasting models and generate 24-month predictions.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting the ML pipeline for sales forecasting..."))

        # Step 1: Train all four models
        metrics = train_and_save_model()

        if not metrics:
            self.stdout.write(self.style.ERROR("Training failed — aborting prediction step."))
            return

        self.stdout.write(self.style.SUCCESS("All models trained successfully."))
        for model_name, m in metrics.items():
            self.stdout.write(
                f"  {model_name:10s}  MAPE={m['mape']:.2f}%  "
                f"RMSE={m['rmse']:,.0f}  R²={m['r2']:.4f}"
            )

        # Step 2: Generate & persist forecasts
        self.stdout.write(self.style.SUCCESS("Generating predictions..."))
        try:
            run_predictions(metrics)
            self.stdout.write(self.style.SUCCESS(
                "Predictions saved to the database. Pipeline complete."
            ))
        except Exception as exc:
            self.stdout.write(self.style.ERROR(f"Prediction step failed: {exc}"))
            raise
