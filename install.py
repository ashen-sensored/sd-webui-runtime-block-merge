import launch

if not launch.is_installed("natsort"):
    launch.run_pip("install natsort", "natsort")
if not launch.is_installed("easing-functions"):
    launch.run_pip("install easing-functions", "easing-functions")