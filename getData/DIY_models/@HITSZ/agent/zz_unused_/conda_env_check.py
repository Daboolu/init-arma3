import subprocess


def get_conda_env_path(env_name):
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            env_list = result.stdout.splitlines()
            for line in env_list:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    path = parts[-1]
                    if name == env_name:
                        return path

            return create_env(env_name)
        else:
            return f"Error occurred: {result.stderr}"

    except Exception as e:
        return f"An error occurred: {e}"


def create_env(env_name):
    try:
        result = subprocess.run(
            ["conda", "create", "--name", env_name, "python=3.8", "--yes"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode == 0:
            return get_conda_env_path(env_name)
        else:
            return f"Error occurred while creating the environment: {result.stderr}"

    except Exception as e:
        return f"An error occurred: {e}"
