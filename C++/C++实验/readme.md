``` mermaid
%%{init: { 'logLevel': 'debug', 'theme': 'base' } }%%
gitGraph
   commit id: "init project"
   commit id: "feat: Add test cases for initial and default poses"
   commit id: "test: Pass initial and default pose test cases"
   commit id: "feat: Add one test cases for move command"
   commit id: "test: Pass move command test cases"
   commit id: "test: Add and Pass all move command test cases"
   commit id: "feat: Add test cases for turn left command"
   commit id: "test: Pass turn left command test cases"
   commit id: "feat: Add test cases for turn right command"
   commit id: "test: Pass turn right command test cases"
   branch lab1-cleancode-base
   commit id: "extract MoveCommand"
   commit id: "extract TurnLeftCommand"
   commit id: "extract TurnRightCommand"
   commit id: "abstract ICommand"
   branch lab2-oop-three-features
   commit id: "feat: Add test cases for fast command"
   commit id: "test: Pass test cases for fast command"
   branch lab2-oop-support-F
   commit id: "command to extract independent file"
   commit id: "extract PoseHandler, decouple ExecutorImpl, Command interdependencies"
   commit id: "command table-driven"
   commit id: "extract Point&Direction, simplify the code cyclomatic complexity in PoseHandler through state changes"
   branch lab3-oop-recfactor-final
   commit id: "use lambda to optimize code"
   commit id: "operator overloading is used to simplify code"
   commit id: "initialization semantics to make the cmderMap init code more concise"
   branch lab3-fp
   commit id: "test:support F&B commander"
   branch lab3-fp-support-B
   commit id: "Singleton factory proivde unified interfaces"
   commit id: "use using feature to optimize code"
   commit id: "introduce action to as real command"
   commit id: "feat: Add test cases for TR command"
   commit id: "test: Pass TR command test cases, TR command easy implement"
   commit id: "physical design and code layering improve code readability"
   branch lab4-comprehensive-actual-combat
   commit id: "feat: Add test cases for Bus&SportCar"
   commit id: "test: Pass Bus&SportCar test cases"
   merge main
```
