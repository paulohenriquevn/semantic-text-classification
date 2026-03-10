"""Unit tests for domain enums."""

import pytest

from talkex.models.enums import (
    Channel,
    ObjectType,
    PoolingStrategy,
    SpeakerRole,
)


class TestSpeakerRole:
    def test_has_expected_members(self) -> None:
        assert set(SpeakerRole) == {
            SpeakerRole.CUSTOMER,
            SpeakerRole.AGENT,
            SpeakerRole.SYSTEM,
            SpeakerRole.UNKNOWN,
        }

    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (SpeakerRole.CUSTOMER, "customer"),
            (SpeakerRole.AGENT, "agent"),
            (SpeakerRole.SYSTEM, "system"),
            (SpeakerRole.UNKNOWN, "unknown"),
        ],
    )
    def test_string_serialization(self, member: SpeakerRole, expected_value: str) -> None:
        assert member.value == expected_value
        assert str(member.value) == expected_value

    def test_is_str_subclass(self) -> None:
        assert isinstance(SpeakerRole.CUSTOMER, str)

    def test_constructs_from_value(self) -> None:
        assert SpeakerRole("customer") is SpeakerRole.CUSTOMER

    def test_rejects_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="is not a valid"):
            SpeakerRole("invalid_role")


class TestChannel:
    def test_has_expected_members(self) -> None:
        assert set(Channel) == {
            Channel.VOICE,
            Channel.CHAT,
            Channel.EMAIL,
            Channel.TICKET,
        }

    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (Channel.VOICE, "voice"),
            (Channel.CHAT, "chat"),
            (Channel.EMAIL, "email"),
            (Channel.TICKET, "ticket"),
        ],
    )
    def test_string_serialization(self, member: Channel, expected_value: str) -> None:
        assert member.value == expected_value

    def test_is_str_subclass(self) -> None:
        assert isinstance(Channel.VOICE, str)

    def test_rejects_invalid_value(self) -> None:
        with pytest.raises(ValueError, match="is not a valid"):
            Channel("sms")


class TestObjectType:
    def test_has_expected_members(self) -> None:
        assert set(ObjectType) == {
            ObjectType.TURN,
            ObjectType.CONTEXT_WINDOW,
            ObjectType.CONVERSATION,
        }

    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (ObjectType.TURN, "turn"),
            (ObjectType.CONTEXT_WINDOW, "context_window"),
            (ObjectType.CONVERSATION, "conversation"),
        ],
    )
    def test_string_serialization(self, member: ObjectType, expected_value: str) -> None:
        assert member.value == expected_value

    def test_is_str_subclass(self) -> None:
        assert isinstance(ObjectType.TURN, str)


class TestPoolingStrategy:
    def test_has_expected_members(self) -> None:
        assert set(PoolingStrategy) == {
            PoolingStrategy.MEAN,
            PoolingStrategy.MAX,
            PoolingStrategy.ATTENTION,
        }

    @pytest.mark.parametrize(
        ("member", "expected_value"),
        [
            (PoolingStrategy.MEAN, "mean"),
            (PoolingStrategy.MAX, "max"),
            (PoolingStrategy.ATTENTION, "attention"),
        ],
    )
    def test_string_serialization(self, member: PoolingStrategy, expected_value: str) -> None:
        assert member.value == expected_value

    def test_is_str_subclass(self) -> None:
        assert isinstance(PoolingStrategy.MEAN, str)


class TestEnumsImportFromModelsPackage:
    """Verify enums are properly re-exported from the models package."""

    def test_speaker_role_importable(self) -> None:
        from talkex.models import SpeakerRole as Imported

        assert Imported is SpeakerRole

    def test_channel_importable(self) -> None:
        from talkex.models import Channel as Imported

        assert Imported is Channel

    def test_object_type_importable(self) -> None:
        from talkex.models import ObjectType as Imported

        assert Imported is ObjectType

    def test_pooling_strategy_importable(self) -> None:
        from talkex.models import PoolingStrategy as Imported

        assert Imported is PoolingStrategy
