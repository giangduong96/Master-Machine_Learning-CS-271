# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/GiangDuong/CLionProjects/q15

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/GiangDuong/CLionProjects/q15/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/q15.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/q15.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/q15.dir/flags.make

CMakeFiles/q15.dir/main.cpp.o: CMakeFiles/q15.dir/flags.make
CMakeFiles/q15.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/GiangDuong/CLionProjects/q15/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/q15.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/q15.dir/main.cpp.o -c /Users/GiangDuong/CLionProjects/q15/main.cpp

CMakeFiles/q15.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/q15.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/GiangDuong/CLionProjects/q15/main.cpp > CMakeFiles/q15.dir/main.cpp.i

CMakeFiles/q15.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/q15.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/GiangDuong/CLionProjects/q15/main.cpp -o CMakeFiles/q15.dir/main.cpp.s

# Object files for target q15
q15_OBJECTS = \
"CMakeFiles/q15.dir/main.cpp.o"

# External object files for target q15
q15_EXTERNAL_OBJECTS =

q15: CMakeFiles/q15.dir/main.cpp.o
q15: CMakeFiles/q15.dir/build.make
q15: CMakeFiles/q15.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/GiangDuong/CLionProjects/q15/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable q15"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/q15.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/q15.dir/build: q15

.PHONY : CMakeFiles/q15.dir/build

CMakeFiles/q15.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/q15.dir/cmake_clean.cmake
.PHONY : CMakeFiles/q15.dir/clean

CMakeFiles/q15.dir/depend:
	cd /Users/GiangDuong/CLionProjects/q15/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/GiangDuong/CLionProjects/q15 /Users/GiangDuong/CLionProjects/q15 /Users/GiangDuong/CLionProjects/q15/cmake-build-debug /Users/GiangDuong/CLionProjects/q15/cmake-build-debug /Users/GiangDuong/CLionProjects/q15/cmake-build-debug/CMakeFiles/q15.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/q15.dir/depend

