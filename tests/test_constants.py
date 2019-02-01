from unittest import TestCase

from src.constants import available_tasks, commands, unknown_class_addition


class TestConstants(TestCase):
    def test_available_tasks(self):
        self.assertEqual(4, len(available_tasks))

    def test_commands(self):
        self.assertEqual(4, len(commands))
        for task in available_tasks:
            self.assertIn(task, commands)
        self.assertEqual(35, len(commands["35-words-recognition"]))
        self.assertEqual(20, len(commands["20-commands-recognition"]))
        self.assertEqual(10, len(commands["10-commands-recognition"]))
        self.assertEqual(2, len(commands["left-right"]))

    def test_unknown_class_addition(self):
        self.assertEqual(4, len(unknown_class_addition))
        for task in available_tasks:
            self.assertIn(task, unknown_class_addition)
        self.assertEqual(bool, type(unknown_class_addition["35-words-recognition"]))
        self.assertEqual(bool, type(unknown_class_addition["20-commands-recognition"]))
        self.assertEqual(bool, type(unknown_class_addition["10-commands-recognition"]))
        self.assertEqual(bool, type(unknown_class_addition["left-right"]))
