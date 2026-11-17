from abc import ABC

from pyromind_common.sdk.utils.models import DiscriminatedUnionMixin


class Animal(DiscriminatedUnionMixin, ABC):
    name: str


class Dog(Animal):
    barking: bool


class Cat(Animal):
    pass
