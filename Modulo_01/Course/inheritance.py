class Vehicle:
    def __init__(self, brand: str, model: str) -> None:
        self.brand = brand
        self.model = model

    def run(self) -> None:
        print(f"Running {self.brand} {self.model}")


class ElectricCar(Vehicle):
    def __init__(self, brand: str, model: str, battery_capacity: float) -> None:
        super().__init__(brand, model)
        self.battery_capacity = battery_capacity
        self.charge: int = 0

    def charge_battery(self, amount: int) -> None:
        self.charge += amount


class CombustionCar(Vehicle):
    def __init__(self, brand: str, model: str, motor_type: str) -> None:
        super().__init__(brand, model)
        self.motorType = motor_type
        self.fuel: float = 0

    def charge_fuel(self, amount: float) -> None:
        self.fuel += amount


print("== Tesla")
tesla = ElectricCar("tesla", "model Tesla", 2000.0)
tesla.charge_battery(100)
print(tesla.brand)
print(tesla.model)
print(tesla.battery_capacity)
print(tesla.charge)
tesla.run()

print("== Subaru")
subaru = CombustionCar("Subaru", "WRX", "boxer")
subaru.charge_fuel(100)
print(subaru.brand)
print(subaru.model)
print(subaru.motorType)
print(subaru.fuel)
subaru.run()
