from planner.person import Person
from planner.queue import Queue
import pytest


def test_queue_create_queue_from_names():
    queue = Queue._create_queue(["Alice", "Bob"])
    assert [person.get_name() for person in queue] == ["Alice", "Bob"]


def test_queue_add_to_queue_adds_new_person_and_updates_existing():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 0)

    assert len(queue.get_list()) == 1

    availability = queue.get_list()[0].get_availability("Monday")

    assert availability[4:6] == [0, 0]

    queue.add_to_queue("Alice", "Monday", "1000", "1030", 1)

    availability = queue.get_list()[0].get_availability("Monday")

    assert availability[4:6] == [0, 0]
    assert availability[6] == 1


def test_queue_add_to_queue_supports_early_start_time():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0800", "1700", 0)

    availability = queue.get_list()[0].get_availability("Monday")

    assert len(availability) == 24

    # 07:00-08:00 is outside the requested availability
    assert availability[:2] == [-1, -1]

    # 08:00-17:00 is available
    assert availability[2:20] == [0] * 18

    # 17:00-19:00 is outside the requested availability
    assert availability[20:] == [-1] * 4


def test_queue_select_available_person_selects_and_moves_to_back():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 0)
    queue.add_to_queue("Bob", "Monday", "0900", "1000", 0)

    selected = queue.select_available_person("Monday", "0900", "0930")
    assert selected is not None
    assert selected.get_name() == "Alice"
    assert queue.get_list()[-1].get_name() == "Alice"
    assert not selected.check_availability("Monday", "0900", "0930")


def test_queue_select_available_person_returns_none_when_no_available_person():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 1)
    assert queue.select_available_person("Monday", "0900", "0930") is None


def test_queue_select_available_person_honors_person_filter():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 0)
    queue.add_to_queue("Bob", "Monday", "0900", "1000", 0)

    selected = queue.select_available_person(
        "Monday",
        "0900",
        "0930",
        person_filter=lambda person: person.get_name() == "Bob"
    )

    assert selected is not None
    assert selected.get_name() == "Bob"
    assert queue.get_list()[-1].get_name() == "Bob"


def test_queue_select_available_person_respects_filter_exclusion():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 0)
    queue.add_to_queue("Bob", "Monday", "0900", "1000", 0)

    selected = queue.select_available_person(
        "Monday",
        "0900",
        "0930",
        person_filter=lambda person: False
    )

    assert selected is None


def test_queue_get_list_returns_internal_queue():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 0)
    assert queue.get_list()[0].get_name() == "Alice"


def test_queue_shuffle_uses_random_shuffle(monkeypatch):
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 0)
    queue.add_to_queue("Bob", "Monday", "0900", "1000", 0)

    captured = []

    def fake_shuffle(lst):
        captured.append(list(lst))
        lst.reverse()

    monkeypatch.setattr("planner.queue.shuffle", fake_shuffle)
    queue.shuffle()

    assert captured
    assert [person.get_name() for person in queue.get_list()] == ["Bob", "Alice"]


def test_queue_find_std_deviation_returns_zero_for_empty_queue():
    queue = Queue()
    assert queue.find_std_deviation() == 0.0


def test_queue_find_std_deviation_computes_correct_value():
    person_a = Person("Alice")
    person_a.set_availability("Monday", "0900", "0930", 1)
    person_a.set_availability("Monday", "0930", "1000", 0)

    person_b = Person("Bob")
    person_b.set_availability("Monday", "0900", "0930", 0)
    person_b.set_availability("Monday", "0930", "1000", 0)

    queue = Queue()
    queue._queue = [person_a, person_b]

    expected_std = 0.25
    assert queue.find_std_deviation() == expected_std


def test_queue_add_to_queue_sets_expected_capacity():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 0, expected_capacity=0.5)

    assert queue.get_list()[0].get_expected_capacity("Monday") == 0.5


def test_queue_add_to_queue_updates_expected_capacity_for_existing_person():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1000", 0, expected_capacity=1.0)
    queue.add_to_queue("Alice", "Tuesday", "0900", "1000", 0, expected_capacity=0.5)

    assert len(queue.get_list()) == 1
    assert queue.get_list()[0].get_expected_capacity("Monday") == 1.0
    assert queue.get_list()[0].get_expected_capacity("Tuesday") == 0.5
    assert queue.get_list()[0].get_expected_capacity() == pytest.approx(0.75)


def test_queue_select_available_person_prefers_higher_expected_capacity():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1200", 0, expected_capacity=0.5)
    queue.add_to_queue("Bob", "Monday", "0900", "1200", 0, expected_capacity=1.0)

    first = queue.select_available_person("Monday", "0900", "0930")
    second = queue.select_available_person("Monday", "0930", "1000")
    third = queue.select_available_person("Monday", "1000", "1030")

    assert first.get_name() == "Alice"
    assert second.get_name() == "Bob"
    assert third.get_name() == "Bob"


def test_queue_find_std_deviation_accounts_for_expected_capacity():
    person_a = Person("Alice", expected_capacity=0.5)
    person_a.set_availability("Monday", "0900", "0930", 1)
    person_a.set_availability("Monday", "0930", "1000", 0)

    person_b = Person("Bob", expected_capacity=1.0)
    person_b.set_availability("Monday", "0900", "0930", 1)
    person_b.set_availability("Monday", "0930", "1000", 0)

    queue = Queue()
    queue._queue = [person_a, person_b]

    # Alice fill ratio 0.5 / capacity 0.5 = 1.0; Bob fill ratio 0.5 / capacity 1.0 = 0.5.
    expected_std = 0.25
    assert queue.find_std_deviation() == expected_std


def test_queue_select_available_person_uses_capacity_for_the_assignment_day():
    queue = Queue()
    queue.add_to_queue("Alice", "Monday", "0900", "1200", 0, expected_capacity=0.5)
    queue.add_to_queue("Alice", "Tuesday", "0900", "1200", 0, expected_capacity=1.0)
    queue.add_to_queue("Bob", "Monday", "0900", "1200", 0, expected_capacity=1.0)
    queue.add_to_queue("Bob", "Tuesday", "0900", "1200", 0, expected_capacity=0.5)

    monday_first = queue.select_available_person("Monday", "0900", "0930")
    monday_second = queue.select_available_person("Monday", "0930", "1000")
    monday_third = queue.select_available_person("Monday", "1000", "1030")

    assert monday_first.get_name() == "Alice"
    assert monday_second.get_name() == "Bob"
    assert monday_third.get_name() == "Bob"

    tuesday_first = queue.select_available_person("Tuesday", "0900", "0930")
    tuesday_second = queue.select_available_person("Tuesday", "0930", "1000")
    tuesday_third = queue.select_available_person("Tuesday", "1000", "1030")

    assert tuesday_first.get_name() == "Alice"
    assert tuesday_second.get_name() == "Bob"
    assert tuesday_third.get_name() == "Alice"
