# Base configuration
ENV_NAME=mia_synthetic
PYTHON_VERSION=3.9

.PHONY: py-install-dev
py-install-dev:
	pip install -e .[dev]

# Generic Commands
define SETUP_ENV
	$(1) create -y -p $(2)/$(ENV_NAME) python=$(PYTHON_VERSION)
endef

define INSTALL_PKG_ENV
	$(1) $(MAKE) py-install-dev
endef

define INIT_JUPYTER
	$(1) pip install ipykernel
	$(1) python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python $(PYTHON_VERSION) ($(ENV_NAME))"
endef

# Conda on Jupyterhub
HUB_MANAGER=conda
HUB_ENV_DIR=/tmp/.conda-envs
HUB_ENV_LOC=$(HUB_ENV_DIR)/$(ENV_NAME)
HUB_RUN_CMD=$(HUB_MANAGER) run -p $(HUB_ENV_LOC)

.PHONY: init-hub-env
init-hub-env:
	$(call SETUP_ENV,$(HUB_MANAGER),$(HUB_ENV_DIR))
	$(HUB_MANAGER) config --append envs_dirs $(HUB_ENV_DIR)

.PHONY: install-pkg-env
install-pkg-env:
	$(call INSTALL_PKG_ENV,$(HUB_RUN_CMD))

.PHONY: init-jupyter
init-jupyter:
	$(call INIT_JUPYTER,$(HUB_RUN_CMD))

.PHONY: set-up-env
set-up-env:
	$(MAKE) init-hub-env
	$(MAKE) install-pkg-env
	$(MAKE) init-jupyter

# Micromamba locally
LOCAL_MANAGER=micromamba
LOCAL_ENV_DIR=~/micromamba/envs
LOCAL_ENV_LOC=$(LOCAL_ENV_DIR)/$(ENV_NAME)
LOCAL_RUN_CMD=$(LOCAL_MANAGER) run -p $(LOCAL_ENV_LOC)

.PHONY: init-local-env
init-local-env:
	$(call SETUP_ENV,$(LOCAL_MANAGER),$(LOCAL_ENV_DIR))

.PHONY: install-pkg-env-local
install-pkg-env-local:
	$(call INSTALL_PKG_ENV,$(LOCAL_RUN_CMD))

.PHONY: init-jupyter-local
init-jupyter-local:
	$(call INIT_JUPYTER,$(LOCAL_RUN_CMD))

.PHONY: set-up-env-local
set-up-env-local:
	$(MAKE) init-local-env
	$(MAKE) install-pkg-env-local
	$(MAKE) init-jupyter-local
