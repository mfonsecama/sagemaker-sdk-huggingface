def plot_results(estimator, metrics="all"):
    """plots tracked result metrics and returns dataframe of it. """
    df = estimator.training_job_analytics.dataframe()
    dx = df.pivot(index="timestamp", columns="metric_name", values="value").reset_index().rename_axis(None, axis=1)
    if metrics == "all":
        dx.plot(kind="line", figsize=(12, 5), x="timestamp")
    else:
        if isinstance(metrics, list):
            metrics.append("timestamp")
            dx[metrics].plot(kind="line", figsize=(12, 5), x="timestamp")
        else:
            dx[[metrics, "timestamp"]].plot(kind="line", figsize=(12, 5), x="timestamp")
    return dx
