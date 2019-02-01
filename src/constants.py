available_tasks = ["35-words-recognition",
                   "20-commands-recognition",
                   "10-commands-recognition",
                   "left-right"]

commands = {"35-words-recognition": ["zero", "one", "two", "three", "four", "five", "six",
                                     "seven", "eight", "nine", "marvin", "sheila", "forward",
                                     "backward", "bed", "bird", "cat", "dog", "down", "follow",
                                     "go", "happy", "house", "learn", "no", "yes", "off",
                                     "on", "right", "left", "stop", "tree", "up", "visual", "wow"],
            "20-commands-recognition": ["yes", "no", "up", "down", "left", "right", "on", "off", "stop",
                                        "go", "zero", "one", "two", "three", "four", "five", "six",
                                        "seven", "eight", "nine"],
            "10-commands-recognition": ["yes", "no", "up", "down", "left", "right", "on", "off",
                                        "stop", "go"],
            "left-right": ["left", "right"]}

unknown_class_addition = {"35-words-recognition": False,
                          "20-commands-recognition": True,
                          "10-commands-recognition": True,
                          "left-right": False}
