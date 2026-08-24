export interface PredictionResponse {
  prediction: number;
  default_probability: number;
  risk_label: 'low_risk' | 'high_risk';
  model_version: string;
  mlflow_run_id: string;
}

export interface PredictionRequest {
  RevolvingUtilizationOfUnsecuredLines: number;
  age: number;
  NumberOfTime30_59DaysPastDueNotWorse: number;
  DebtRatio: number;
  MonthlyIncome?: number;
  NumberOfOpenCreditLinesAndLoans: number;
  NumberOfTimes90DaysLate: number;
  NumberRealEstateLoansOrLines: number;
  NumberOfTime60_89DaysPastDueNotWorse: number;
  NumberOfDependents?: number;
}
