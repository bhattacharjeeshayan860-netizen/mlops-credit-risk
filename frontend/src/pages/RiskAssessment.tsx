import { useState } from 'react';
import axios from 'axios';
import { Loader2, AlertCircle, RefreshCw, ShieldCheck } from 'lucide-react';
import type { PredictionRequest, PredictionResponse } from '../types/prediction';
import RiskResult from './components/RiskResult';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export default function RiskAssessment() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<PredictionRequest>({
    RevolvingUtilizationOfUnsecuredLines: 0,
    age: 30,
    NumberOfTime30_59DaysPastDueNotWorse: 0,
    DebtRatio: 0,
    MonthlyIncome: 0,
    NumberOfOpenCreditLinesAndLoans: 0,
    NumberOfTimes90DaysLate: 0,
    NumberRealEstateLoansOrLines: 0,
    NumberOfTime60_89DaysPastDueNotWorse: 0,
    NumberOfDependents: 0,
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    const val = e.target.type === 'number' ? parseFloat(value) : value;
    setFormData((prev) => ({ ...prev, [name]: val }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post<PredictionResponse>(`${API_URL}/predict`, formData);
      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
    setFormData({
      RevolvingUtilizationOfUnsecuredLines: 0,
      age: 30,
      NumberOfTime30_59DaysPastDueNotWorse: 0,
      DebtRatio: 0,
      MonthlyIncome: 0,
      NumberOfOpenCreditLinesAndLoans: 0,
      NumberOfTimes90DaysLate: 0,
      NumberRealEstateLoansOrLines: 0,
      NumberOfTime60_89DaysPastDueNotWorse: 0,
      NumberOfDependents: 0,
    });
  };

  return (
    <div className="max-w-6xl mx-auto p-6 lg:p-10">
      <header className="mb-8 flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
        <div className="space-y-2">
          <div className="mb-2 flex items-center gap-2">
            <Badge variant="teal">Inference Engine</Badge>
            <span className="text-xs font-medium text-slate-400">Live production environment</span>
          </div>
          <h1 className="text-4xl font-black tracking-tight text-slate-900">Risk Assessment</h1>
          <p className="max-w-2xl text-lg text-slate-500">
            Generate real-time credit delinquency estimates with the current production model.
          </p>
        </div>

        <div className="flex w-full flex-col gap-3 md:w-auto md:flex-row">
          <Button type="button" variant="outline" onClick={handleReset} className="w-full md:w-auto">
            <RefreshCw size={16} className="mr-2" />
            Reset
          </Button>
          <Button type="submit" onClick={handleSubmit} disabled={loading} size="lg" className="w-full md:w-auto px-8 shadow-lg shadow-teal-500/15">
            {loading ? <Loader2 className="mr-2 animate-spin" size={18} /> : null}
            {loading ? 'Assessing...' : 'Run Assessment'}
          </Button>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-10 lg:grid-cols-12">
        <div className="space-y-8 lg:col-span-8">
          <form onSubmit={handleSubmit} className="space-y-8">
            <Card header="Customer Profile">
              <div className="grid grid-cols-1 gap-x-8 gap-y-6 md:grid-cols-2">
                <div className="space-y-1.5">
                  <Input label="Age" type="number" name="age" value={formData.age} onChange={handleChange} placeholder="e.g. 35" required />
                  <p className="text-[11px] text-slate-400">Applicant's current age.</p>
                </div>

                <div className="space-y-1.5">
                  <Input label="Number of Dependents" type="number" name="NumberOfDependents" value={formData.NumberOfDependents} onChange={handleChange} placeholder="0" />
                  <p className="text-[11px] text-slate-400">Dependents supported by the applicant.</p>
                </div>

                <div className="space-y-1.5 md:col-span-2">
                  <Input label="Monthly Income ($)" type="number" name="MonthlyIncome" value={formData.MonthlyIncome} onChange={handleChange} placeholder="e.g. 5000" />
                  <p className="text-[11px] text-slate-400">Verified net monthly income.</p>
                </div>
              </div>
            </Card>

            <Card header="Credit Utilization & Debt">
              <div className="grid grid-cols-1 gap-x-8 gap-y-6 md:grid-cols-2">
                <div className="space-y-1.5">
                  <Input label="Revolving Utilization" type="number" step="0.01" name="RevolvingUtilizationOfUnsecuredLines" value={formData.RevolvingUtilizationOfUnsecuredLines} onChange={handleChange} placeholder="0.00" required />
                  <p className="text-[11px] text-slate-400">Ratio of total credit used to available credit.</p>
                </div>

                <div className="space-y-1.5">
                  <Input label="Debt Ratio" type="number" step="0.01" name="DebtRatio" value={formData.DebtRatio} onChange={handleChange} placeholder="0.00" required />
                  <p className="text-[11px] text-slate-400">Monthly obligations as a share of income.</p>
                </div>

                <div className="space-y-1.5">
                  <Input label="Open Credit Lines" type="number" name="NumberOfOpenCreditLinesAndLoans" value={formData.NumberOfOpenCreditLinesAndLoans} onChange={handleChange} placeholder="0" required />
                  <p className="text-[11px] text-slate-400">Total active credit accounts.</p>
                </div>

                <div className="space-y-1.5">
                  <Input label="Real Estate Loans" type="number" name="NumberRealEstateLoansOrLines" value={formData.NumberRealEstateLoansOrLines} onChange={handleChange} placeholder="0" required />
                  <p className="text-[11px] text-slate-400">Mortgages or home equity lines.</p>
                </div>
              </div>
            </Card>

            <Card header="Payment History">
              <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                <div className="space-y-1.5">
                  <Input label="30–59 Days Late" type="number" name="NumberOfTime30_59DaysPastDueNotWorse" value={formData.NumberOfTime30_59DaysPastDueNotWorse} onChange={handleChange} placeholder="0" required />
                </div>
                <div className="space-y-1.5">
                  <Input label="60–89 Days Late" type="number" name="NumberOfTime60_89DaysPastDueNotWorse" value={formData.NumberOfTime60_89DaysPastDueNotWorse} onChange={handleChange} placeholder="0" required />
                </div>
                <div className="space-y-1.5">
                  <Input label="90+ Days Late" type="number" name="NumberOfTimes90DaysLate" value={formData.NumberOfTimes90DaysLate} onChange={handleChange} placeholder="0" required />
                </div>
              </div>
            </Card>
          </form>
        </div>

        <div className="lg:col-span-4">
          <div className="sticky top-8 space-y-6">
            {error && (
              <div className="flex items-start gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-red-700">
                <AlertCircle size={18} className="mt-0.5 shrink-0" />
                <div className="space-y-1">
                  <p className="text-sm font-bold">Inference failed</p>
                  <p className="text-xs opacity-90">{error}</p>
                </div>
              </div>
            )}

            {result ? (
              <RiskResult result={result} />
            ) : (
              <div className="flex min-h-[420px] flex-col items-center justify-center space-y-6 rounded-2xl border-2 border-dashed border-slate-200 bg-white p-10 text-center text-slate-400">
                <div className="flex h-20 w-20 items-center justify-center rounded-full bg-slate-100">
                  <ShieldCheck size={38} className="text-slate-300" />
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-bold text-slate-600">Ready for assessment</p>
                  <p className="text-xs leading-relaxed text-slate-500">Complete the applicant profile to produce a real-time risk evaluation.</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

