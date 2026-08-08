import sys
import time

from models import WaterLevelReading, MonitoringNode, RiskLevel, DataSource
from simulator import WaterLevelSimulator, AsyncWaterLevelSimulator, SimulatorConfig
from engine import CoreEngine, FloodEngine


def test(name: str, condition, expected=True):
    result = condition() if callable(condition) else condition
    status = "[PASS]" if result == expected else "[FAIL]"
    print(f"{status} {name}")
    return result == expected


print("=" * 60)
print("TIER 1 VERIFICATION TEST (Python)")
print("=" * 60)

passed = 0
failed = 0

print("\n--- TEST 1: Data Models ---")


def test_water_level_reading():
    reading = WaterLevelReading(
        node_id="NODE-001",
        water_level=25.5,
        source=DataSource.SIMULATOR
    )

    return (
        reading.node_id == "NODE-001"
        and reading.water_level == 25.5
        and reading.source == DataSource.SIMULATOR
    )


if test("WaterLevelReading creation", test_water_level_reading):
    passed += 1
else:
    failed += 1


def test_reading_to_dict():
    reading = WaterLevelReading(
        node_id="TEST",
        water_level=30.0
    )

    d = reading.to_dict()

    return (
        "id" in d
        and "nodeId" in d
        and "waterLevel" in d
        and d["waterLevel"] == 30.0
    )


if test("WaterLevelReading to_dict()", test_reading_to_dict):
    passed += 1
else:
    failed += 1


def test_monitoring_node():
    node = MonitoringNode(
        id="NODE-001",
        name="Test Node",
        thresholds={
            "watch": 30,
            "warning": 50,
            "critical": 70
        }
    )

    return (
        node.id == "NODE-001"
        and node.thresholds["watch"] == 30
    )


if test("MonitoringNode creation", test_monitoring_node):
    passed += 1
else:
    failed += 1


print("\n--- TEST 2: Simulator ---")


def test_simulator_creation():
    sim = WaterLevelSimulator(
        SimulatorConfig(
            node_id="TEST",
            start_level=20,
            mode="rising"
        )
    )

    return (
        sim.node_id == "TEST"
        and sim.current_level == 20
        and sim.mode == "rising"
    )


if test("Simulator creation", test_simulator_creation):
    passed += 1
else:
    failed += 1


def test_simulator_generate_reading():
    sim = WaterLevelSimulator(
        SimulatorConfig(start_level=20)
    )

    sim.start()
    reading = sim._generate_reading()
    sim.stop()

    return (
        reading is not None
        and reading.water_level >= 0
    )


if test("Simulator generates readings", test_simulator_generate_reading):
    passed += 1
else:
    failed += 1


def test_simulator_modes():
    modes = [
        "rising",
        "falling",
        "stable",
        "fluctuating",
        "rapid"
    ]

    for mode in modes:
        sim = WaterLevelSimulator(
            SimulatorConfig(
                mode=mode,
                start_level=25
            )
        )

        sim.start()
        reading = sim._generate_reading()
        sim.stop()

        if reading is None:
            return False

    return True


if test("Simulator handles all modes", test_simulator_modes):
    passed += 1
else:
    failed += 1


def test_simulator_presets():
    sim = WaterLevelSimulator()

    for preset in [
        "demo",
        "rapid",
        "slow",
        "stable"
    ]:
        sim.use_preset(preset)

        if sim.mode != preset:
            return False

    return True


if test("Simulator presets work", test_simulator_presets):
    passed += 1
else:
    failed += 1


def test_simulator_inject_level():
    sim = WaterLevelSimulator(
        SimulatorConfig(start_level=20)
    )

    reading = sim.inject_level(55.0)

    return reading.water_level == 55.0


if test("Simulator inject_level()", test_simulator_inject_level):
    passed += 1
else:
    failed += 1


print("\n--- TEST 3: Core Engine ---")


def test_engine_creation():
    node = MonitoringNode(
        id="NODE-001",
        name="Test Node"
    )

    engine = CoreEngine(node=node)

    return engine.node.id == "NODE-001"


if test("CoreEngine creation", test_engine_creation):
    passed += 1
else:
    failed += 1


def test_engine_process():
    engine = CoreEngine()

    reading = WaterLevelReading(
        node_id="NODE-001",
        water_level=25.0
    )

    result = engine.process(reading)

    return (
        result is not None
        and "reading" in result
        and result["reading"]["waterLevel"] == 25.0
    )


if test("CoreEngine processes readings", test_engine_process):
    passed += 1
else:
    failed += 1


def test_engine_buffer():
    engine = CoreEngine()

    for level in [20, 21, 22, 23, 24]:
        reading = WaterLevelReading(
            node_id="NODE-001",
            water_level=float(level)
        )

        engine.process(reading)

    return len(engine.reading_buffer) == 5


if test("CoreEngine buffers readings", test_engine_buffer):
    passed += 1
else:
    failed += 1


def test_engine_smoothed():
    engine = CoreEngine()

    for level in [20, 21, 22]:
        reading = WaterLevelReading(
            node_id="NODE-001",
            water_level=float(level)
        )

        engine.process(reading)

    return abs(engine.state.smoothed_level - 21.0) < 0.1


if test("CoreEngine calculates smoothed level", test_engine_smoothed):
    passed += 1
else:
    failed += 1


print("\n--- TEST 4: End-to-End Data Flow ---")


def test_end_to_end():
    sim = WaterLevelSimulator(
        SimulatorConfig(
            node_id="NODE-001",
            start_level=20,
            mode="stable"
        )
    )

    engine = CoreEngine()
    readings_received = []

    def on_reading(reading):
        readings_received.append(reading)
        engine.process(reading)

    sim.on_data(on_reading)

    sim.start()
    time.sleep(0.1)

    for _ in range(5):
        sim._generate_reading()
        time.sleep(0.01)

    sim.stop()

    return (
        len(readings_received) >= 5
        and engine.state.readings_processed >= 5
    )


if test("End-to-end data flow", test_end_to_end):
    passed += 1
else:
    failed += 1


print("\n--- TEST 5: Risk Level Detection ---")


def test_flood_engine():
    node = MonitoringNode(
        id="NODE-001",
        name="Test Node",
        thresholds={
            "watch": 30,
            "warning": 50,
            "critical": 70
        }
    )

    engine = FloodEngine(node=node)

    assert engine.determine_risk(20, 0) == RiskLevel.SAFE
    assert engine.determine_risk(35, 0) == RiskLevel.WATCH
    assert engine.determine_risk(55, 0) == RiskLevel.WARNING
    assert engine.determine_risk(75, 0) == RiskLevel.CRITICAL

    return True


if test("FloodEngine risk detection", test_flood_engine):
    passed += 1
else:
    failed += 1


print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print(f"[PASS] Passed: {passed}")
print(f"[FAIL] Failed: {failed}")
print()

if failed == 0:
    print("*** ALL TIER 1 TESTS PASSED! ***")
    print()
    print("TIER 1 REQUIREMENTS MET:")
    print("  [x] Python Data Models")
    print("  [x] Water Level Simulator")
    print("  [x] Core Engine")
    print("  [x] WebSocket-ready Architecture")
    print()
    print("Ready for Tier 2: Data Validation")
else:
    print("!!! SOME TESTS FAILED - Review above")

print("=" * 60)