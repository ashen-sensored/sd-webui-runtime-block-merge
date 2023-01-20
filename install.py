import launch

if not launch.is_installed("natsort"):
    launch.run_pip("install natsort", "natsort")