

export default function BenchmarkPage() {
  return (
    <div className="animate-fade-in">
      <h1>Benchmark Runs</h1>
      <p>Execute benchmarks across multiple evaluation questions and store the results.</p>
      <div className="card">
        <p>Run benchmarking scripts from the CLI or click below to trigger a small batch.</p>
        <button className="btn btn-primary mt-4">Run Benchmark Batch</button>
      </div>
    </div>
  );
}
