OUT=bin
SOURCE=src
WRAP=wrapper
TESTER=tester
CXX=g++
CXXFLAGS=-Wall -pedantic -Wextra -Wno-long-long -O3 -std=c++11 -D __verbose__
LD=g++
# pybind11
# g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) wrapper/rbc.cpp -o rbc$(python3-config --extension-suffix)

all: compile

run: compile
	@echo "> Running tests!"
	$(SOURCE)/tester.sh

clean:
	@echo "> Cleaning project"
	rm -rf $(OUT) $(TESTER)

compile: $(OUT) $(OUT)/$(TESTER)
	cp "$(OUT)/$(TESTER)" "$(TESTER)"

$(OUT)/$(TESTER): $(OUT)/utils.o $(OUT)/logger.o $(OUT)/ruleset.o\
 $(OUT)/rule_learner.o $(OUT)/tester.o
	$(LD) $^ -o $@

$(OUT):
	@echo "> Creating $@"
	@mkdir -p "$@"

$(OUT)/%o: $(SOURCE)/%cpp
	@echo "> Creating $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OUT)/utils.o: $(SOURCE)/utils.cpp $(SOURCE)/utils.hpp $(SOURCE)/ruleset.hpp
$(OUT)/logger.o: $(SOURCE)/logger.cpp $(SOURCE)/logger.hpp
$(OUT)/ruleset.o: $(SOURCE)/ruleset.cpp $(SOURCE)/ruleset.hpp $(SOURCE)/logger.hpp
$(OUT)/rule_learner.o: $(SOURCE)/rule_learner.cpp $(SOURCE)/rule_learner.hpp\
 $(SOURCE)/ruleset.hpp $(SOURCE)/logger.hpp $(SOURCE)/utils.hpp
$(OUT)/tester.o: $(SOURCE)/tester.cpp $(SOURCE)/ruleset.hpp\
 $(SOURCE)/rule_learner.hpp
