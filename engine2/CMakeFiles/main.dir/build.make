# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.27.6/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.27.6/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/src/layer.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/layer.cpp.o: src/layer.cpp
CMakeFiles/main.dir/src/layer.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/src/layer.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/layer.cpp.o -MF CMakeFiles/main.dir/src/layer.cpp.o.d -o CMakeFiles/main.dir/src/layer.cpp.o -c /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/layer.cpp

CMakeFiles/main.dir/src/layer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/layer.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/layer.cpp > CMakeFiles/main.dir/src/layer.cpp.i

CMakeFiles/main.dir/src/layer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/layer.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/layer.cpp -o CMakeFiles/main.dir/src/layer.cpp.s

CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/main.cpp.o: src/main.cpp
CMakeFiles/main.dir/src/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/src/main.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/main.cpp.o -MF CMakeFiles/main.dir/src/main.cpp.o.d -o CMakeFiles/main.dir/src/main.cpp.o -c /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/main.cpp

CMakeFiles/main.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/main.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/main.cpp > CMakeFiles/main.dir/src/main.cpp.i

CMakeFiles/main.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/main.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/main.cpp -o CMakeFiles/main.dir/src/main.cpp.s

CMakeFiles/main.dir/src/net.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/net.cpp.o: src/net.cpp
CMakeFiles/main.dir/src/net.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/src/net.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/net.cpp.o -MF CMakeFiles/main.dir/src/net.cpp.o.d -o CMakeFiles/main.dir/src/net.cpp.o -c /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/net.cpp

CMakeFiles/main.dir/src/net.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/net.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/net.cpp > CMakeFiles/main.dir/src/net.cpp.i

CMakeFiles/main.dir/src/net.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/net.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/net.cpp -o CMakeFiles/main.dir/src/net.cpp.s

CMakeFiles/main.dir/src/neuron.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/neuron.cpp.o: src/neuron.cpp
CMakeFiles/main.dir/src/neuron.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/main.dir/src/neuron.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/neuron.cpp.o -MF CMakeFiles/main.dir/src/neuron.cpp.o.d -o CMakeFiles/main.dir/src/neuron.cpp.o -c /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/neuron.cpp

CMakeFiles/main.dir/src/neuron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/neuron.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/neuron.cpp > CMakeFiles/main.dir/src/neuron.cpp.i

CMakeFiles/main.dir/src/neuron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/neuron.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/src/neuron.cpp -o CMakeFiles/main.dir/src/neuron.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/src/layer.cpp.o" \
"CMakeFiles/main.dir/src/main.cpp.o" \
"CMakeFiles/main.dir/src/net.cpp.o" \
"CMakeFiles/main.dir/src/neuron.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/src/layer.cpp.o
main: CMakeFiles/main.dir/src/main.cpp.o
main: CMakeFiles/main.dir/src/net.cpp.o
main: CMakeFiles/main.dir/src/neuron.cpp.o
main: CMakeFiles/main.dir/build.make
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2 /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2 /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2 /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2 /Users/rae/Documents/GithubDesktopProjects/MNISIT-CPP/engine2/CMakeFiles/main.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/main.dir/depend
