
#############################
# ----- CONFIGURATION ----- #
#############################

PARAM_SEQ_LEN ?= 32
PARAM_INPUT_DIM ?= 64

#####################
# ----- RULES ----- #
#####################

# Default alias
.PHONY: all
all: data.h

# Generate data.h
data.h: .force
	@echo "## Generating data.h..."
	PYTHONPATH=../../../nmc python3 datagen.py --seq_len $(PARAM_SEQ_LEN) --input_dim $(PARAM_INPUT_DIM)

# Dummy rule to force recompilation (because of parameters)
.force:

# Clean
.PHONY: clean
clean:
	$(RM) data.h