module.exports = {
  apps: [
    {
      name: "face-detection-backend",
      script: "uvicorn",
      args: "main:app --host 0.0.0.0 --port 8000",
      cwd: "./backend",
      interpreter: "python3",
      instances: 4,
      exec_mode: "cluster",
      watch: false,
      max_memory_restart: "2G",
      env: {
        DETECTOR_METHOD: "haar",
        REDIS_HOST: "localhost",
      },
    },
    {
      name: "face-detection-frontend",
      script: "serve",
      args: "-s build -l 3000",
      cwd: "./frontend",
      instances: 1,
      exec_mode: "fork",
    },
  ],
};
