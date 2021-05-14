from random import randint
import pandas as pd


class Serializable:
    """When we want a custom object to be serializable within the Syft ecosystem
    (as outline in the tutorial above), the first thing we need to do is have it
    subclass from this class. You must then do the following in order for the
    subclass to be properly implemented:

    - implement a protobuf file in the "PySyft/proto" folder for this custom class.
    - compile the protobuf file by running `bash scripts/build_proto`
    - find the generated python file in syft.proto
    - import the generated protobuf class into my custom class
    - implement get_protobuf_schema
    - implement <my class>._object2proto() method to serialize the object to protobuf
    - implement <my class>._proto2object() to deserialize the protobuf object

    At this point, your class should be ready to serialize and deserialize! Don't
    forget to add tests for your object!

    If you want to wrap an existing type (like a torch.tensor) to be used in our serialization
    ecosystem, you should consider wrapping it. Wrapping means that we NEVER use the wrapper
    further more into our ecosystem, we only need an easy interface to serialize wrappers.

    Eg:

    .. code-block:: python

        class WrapperInt(Serializable)
            def __init__(self, value: int):
               self.int_obj = value

            def _object2proto(self):
               ...

            @staticmethod
            def _proto2object(proto):
               ...

            @staticmethod
            def get_protobuf_schema():
               ...

            @staticmethod
            def get_wrapped_type():
               return int

    You must implement the following in order for the subclass to be properly implemented to be
    seen as a wrapper:

    - everything presented in the first tutorial of this docstring.
    - implement get_wrapped_type to return the wrapped type.

    Note: A wrapper should NEVER be used in the codebase, these are only for serialization purposes
    on external objects.

    After doing all of the above steps, you can call something like sy.serialize(5) and be
    serialized using our messaging proto backbone.
    """

    @property
    def named(self):
        if hasattr(self, "name"):
            return self.name  # type: ignore
        else:
            return "UNNAMED"

    @property
    def class_name(self):
        return str(self.__class__.__name__)

    @property
    def icon(self):
        # as in cereal, get it!?
        return "🌾"

    @property
    def pprint(self):
        return f"{self.icon} {self.named} ({self.class_name})"

    @staticmethod
    def random_name():
        return f'random_name_${randint(0, 1000)}'


class StoreClient:
    def __init__(self, client) -> None:
        self.client = client

    @property
    def store(self):
        # msg = ObjectSearchMessage(
        #     address=self.client.address, reply_to=self.client.address
        # )
        msg = "asd"

        results = getattr(
            self.client.send_immediate_msg_with_reply(msg=msg), "results", None
        )
        if results is None:
            raise ValueError("TODO")

        # This is because of a current limitation in Pointer where we cannot
        # serialize a client object. TODO: Fix limitation in Pointer so that we don't need this.
        for result in results:
            result.gc_enabled = False
            result.client = self.client

        return results

    def __len__(self) -> int:
        """Return the number of items in the object store we're allowed to know about"""

        return len(self.store)

    def __getitem__(self, key):
        if isinstance(key, str):
            matches = 0
            match_obj = None

            for obj in self.store:
                if key in obj.tags:
                    matches += 1
                    match_obj = obj
            if matches == 1 and match_obj is not None:
                return match_obj
            elif matches > 1:
                raise KeyError("More than one item with tag:" + str(key))
            else:
                # If key does not math with any tags, we then try to match it with id string.
                # But we only do this if len(key)>=5, because if key is too short, for example
                # if key="a", there are chances of mismatch it with id string, and I don't
                # think the user pass a key such short as part of id string.
                if len(key) >= 5:
                    for obj in self.store:
                        if key in str(obj.id_at_location.value).replace("-", ""):
                            return obj
                else:
                    raise KeyError(f"No such item found for tag: {key}, and we "
                                   + "don't consider it as part of id string because its too short."
                                   )

            raise KeyError("No such item found for id:" + str(key))
        if isinstance(key, int):
            return self.store[key]
        else:
            raise KeyError("Please pass in a string or int key")

    def __repr__(self) -> str:
        return repr(self.store)

    @property
    def pandas(self) -> pd.DataFrame:
        obj_lines = list()
        for obj in self.store:
            obj_lines.append(
                {
                    "ID": obj.id_at_location,
                    "Tags": obj.tags,
                    "Description": obj.description,
                    "object_type": obj.object_type,
                }
            )
        return pd.DataFrame(obj_lines)


class Address(Serializable):
    name = None

    def __init__(
        self,
        name=None,
        network=None,
        domain=None,
        device=None,
        vm=None,
    ):
        self.name = name if name is not None else Serializable.random_name()

        # this address points to a node, if that node lives within a network,
        # or is a network itself, this property will store the ID of that network
        # if it is known.
        self._network = network

        # this address points to a node, if that node lives within a domain
        # or is a domain itself, this property will store the ID of that domain
        # if it is known.
        self._domain = domain

        # this address points to a node, if that node lives within a device
        # or is a device itself, this property will store the ID of that device
        # if it is known
        self._device = device

        # this client points to a node, if that node lives within a vm
        # or is a vm itself, this property will store the ID of that vm if it
        # is known
        self._vm = vm

    @property
    def icon(self):
        # 4 different aspects of location
        icon = "💠"
        sub = []
        if self.vm is not None:
            sub.append("🍰")
        if self.device is not None:
            sub.append("📱")
        if self.domain is not None:
            sub.append("🏰")
        if self.network is not None:
            sub.append("🔗")

        if len(sub) > 0:
            icon = f"{icon} ["
            for s in sub:
                icon += s
            icon += "]"
        return icon

    @property
    def pprint(self):
        output = f"{self.icon} {self.named} ({self.class_name})"
        # if hasattr(self, "id"):
        #     output += f"@{self.target_id.id.emoji()}"
        return output

    def post_init(self):
        print(f"> Creating {self.pprint}")

    def key_emoji(self, key):
        return "ALL"

    @property
    def address(self):
        # QUESTION what happens if we have none of these?

        # sneak the name on there
        if hasattr(self, "name"):
            name = self.name
        else:
            name = Serializable.random_name()

        address = Address(
            name=name,
            network=self.network,
            domain=self.domain,
            device=self.device,
            vm=self.vm,
        )

        return address

    @property
    def network(self):
        """This client points to a node, if that node lives within a network
        or is a network itself, this property will return the ID of that network
        if it is known by the client."""

        return self._network

    @network.setter
    def network(self, new_network):
        """This client points to a node, if that node lives within a network
        or is a network itself and we learn the id of that network, this setter
        allows us to save the id of that network for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages. That
        address object will include this information if it is available"""
        self._network = new_network
        return self._network

    @property
    def network_id(self):
        network = self.network
        if network is not None:
            return network.id
        return None

    @property
    def domain(self):
        """This client points to a node, if that node lives within a domain
        or is a domain itself, this property will return the ID of that domain
        if it is known by the client."""

        return self._domain

    @domain.setter
    def domain(self, new_domain):
        """This client points to a node, if that node lives within a domain
        or is a domain itself and we learn the id of that domain, this setter
        allows us to save the id of that domain for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._domain = new_domain
        return self._domain

    @property
    def domain_id(self):
        domain = self.domain
        if domain is not None:
            return domain.id
        return None

    @property
    def device(self):
        """This client points to a node, if that node lives within a device
        or is a device itself, this property will return the ID of that device
        if it is known by the client."""
        return self._device

    @device.setter
    def device(self, new_device):
        """This client points to a node, if that node lives within a device
        or is a device itself and we learn the id of that device, this setter
        allows us to save the id of that device for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._device = new_device
        return self._device

    @property
    def device_id(self):
        device = self.device
        if device is not None:
            return device.id
        return None

    @property
    def vm(self):
        """This client points to a node, if that node lives within a vm
        or is a vm itself, this property will return the ID of that vm
        if it is known by the client."""

        return self._vm

    @vm.setter
    def vm(self, new_vm):
        """This client points to a node, if that node lives within a vm
        or is a vm itself and we learn the id of that vm, this setter
        allows us to save the id of that vm for use later. We use a getter
        (@property) and setter (@set) explicitly because we want all clients
        to efficiently save an address object for use when sending messages to their
        target. That address object will include this information if it is available"""

        self._vm = new_vm
        return self._vm

    @property
    def vm_id(self):
        vm = self.vm
        if vm is not None:
            return vm.id
        return None

    def target_emoji(self):
        output = ""
        if self.target_id is not None:
            output = f"@{self.target_id.id.emoji()}"
        return output

    @property
    def target_id(self):
        """Return the address of the node which lives at this address.

        Note that this id is simply the most granular id available to the
        address."""
        if self._vm is not None:
            return self._vm
        elif self._device is not None:
            return self._device
        elif self._domain is not None:
            return self._domain
        elif self._network is not None:
            return self._network

    def __eq__(self, other):
        """Returns whether two Address objects refer to the same set of locations

        :param other: the other object to compare with self
        :type other: Any (note this must be Any or __eq__ fails on other types)
        :returns: whether the two objects are the same
        :rtype: bool
        """

        try:
            a = self.network == other.network
            b = self.domain == other.domain
            c = self.device == other.device
            d = self.vm == other.vm

            return a and b and c and d
        except Exception:
            return False

    def __repr__(self):
        out = f"<{type(self).__name__} -"
        if self.network is not None:
            out += f" Network:{self.network.repr_short()},"  # OpenGrid
        if self.domain is not None:
            out += f" Domain:{self.domain.repr_short()} "  # UCSF
        if self.device is not None:
            out += f" Device:{self.device.repr_short()},"  # One of UCSF's Dell Servers
        if self.vm is not None:
            out += f" VM:{self.vm.repr_short()}"  # 8GB RAM set aside @Trask - UCSF-Server-5

        # remove extraneous comma and add a close carrot
        return out[:-1] + ">"


class AbstractNode(Address):

    name = None
    signing_key = None
    verify_key = None
    root_verify_key = None
    guest_verify_key_registry = None
    admin_verify_key_registry = None
    cpl_ofcr_verify_key_registry = None

    def __init__(
        self,
        name=None,
        network=None,
        domain=None,
        device=None,
        vm=None,
    ):
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

    store = None
    requests = None
    lib_ast = None  # Can't import Globals (circular reference)
    """"""

    @property
    def keys(self) -> str:
        verify = (
            self.key_emoji(key=self.signing_key.verify_key)
            if self.signing_key is not None
            else "🚫"
        )
        root = (
            self.key_emoji(key=self.root_verify_key)
            if self.root_verify_key is not None
            else "🚫"
        )
        keys = f"🔑 {verify}" + f"🗝 {root}"

        return keys


class AbstractNodeClient(Address):
    lib_ast = None  # Can't import Globals (circular reference)
    # TODO: remove hacky in_memory_client_registry
    in_memory_client_registry = None
    """"""

    @property
    def id(self):
        """This client points to an node, this returns the id of that node."""
        raise NotImplementedError

    def send_immediate_msg_without_reply(self, msg):
        raise NotImplementedError


class Client(AbstractNodeClient):
    """Client is an incredibly powerful abstraction in Syft. We assume that,
    no matter where a client is, it can figure out how to communicate with
    the Node it is supposed to point to. If I send you a client I have
    with all of the metadata in it, you should have all the information
    you need to know to interact with a node (although you might not
    have permissions - clients should not store private keys)."""

    def __init__(
        self,
        name,
        routes,
        network=None,
        domain=None,
        device=None,
        vm=None,
        signing_key=None,
        verify_key=None,
    ):
        name = f"{name} Client" if name is not None else None
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        self.routes = routes
        self.default_route_index = 0

        self.store = StoreClient(client=self)

    @property
    def icon(self):
        icon = "📡"
        sub = []
        if self.vm is not None:
            sub.append("🍰")
        if self.device is not None:
            sub.append("📱")
        if self.domain is not None:
            sub.append("🏰")
        if self.network is not None:
            sub.append("🔗")

        if len(sub) > 0:
            icon = f"{icon} ["
            for s in sub:
                icon += s
            icon += "]"
        return icon

    # TODO fix the msg type but currently tensor needs SyftMessage

    def send_immediate_msg_with_reply(
        self,
        msg,
        route_index: int = 0,
    ):
        route_index = route_index or self.default_route_index

        response = self.routes[route_index].send_immediate_msg_with_reply(msg=msg)
        return response.message

    # TODO fix the msg type but currently tensor needs SyftMessage

    def send_immediate_msg_without_reply(
        self,
        msg,
        route_index: int = 0,
    ):
        route_index = route_index or self.default_route_index
        self.routes[route_index].send_immediate_msg_without_reply(msg=msg)

    def send_eventual_msg_without_reply(
        self, msg, route_index: int = 0
    ):
        route_index = route_index or self.default_route_index

        self.routes[route_index].send_eventual_msg_without_reply(msg=msg)

    def __repr__(self):
        return f"<Client pointing to node with id:{self.id}>"

    def register_route(self, route):
        self.routes.append(route)

    def set_default_route(self, route_index: int):
        self.default_route = route_index

    @property
    def keys(self):
        verify = (
            self.key_emoji(key=self.signing_key.verify_key)
            if self.signing_key is not None
            else "🚫"
        )
        keys = f"🔑 {verify}"

        return keys

    def __hash__(self):
        return hash(self.id)


class BidirectionalConnection(object):
    def recv_immediate_msg_with_reply(
        self, msg
    ):
        raise NotImplementedError


class ServerConnection(object):
    def __init__(self) -> None:
        self.opt_bidirectional_conn = BidirectionalConnection()


class ClientConnection(object):
    def __init__(self) -> None:
        self.opt_bidirectional_conn = BidirectionalConnection()


class VirtualServerConnection(ServerConnection):
    def __init__(self, node: AbstractNode):
        self.node = node

    def recv_immediate_msg_with_reply(
        self, msg
    ):
        return self.node.recv_immediate_msg_with_reply(msg=msg)

    def recv_immediate_msg_without_reply(
        self, msg
    ) -> None:
        self.node.recv_immediate_msg_without_reply(msg=msg)

    def recv_eventual_msg_without_reply(
        self, msg
    ) -> None:
        self.node.recv_eventual_msg_without_reply(msg=msg)


class VirtualClientConnection(ClientConnection):
    def __init__(self, server: VirtualServerConnection):
        self.server = server

    def send_immediate_msg_without_reply(
        self, msg
    ) -> None:
        self.server.recv_immediate_msg_without_reply(msg=msg)

    def send_immediate_msg_with_reply(
        self, msg
    ):
        return self.server.recv_immediate_msg_with_reply(msg=msg)

    def send_eventual_msg_without_reply(
        self, msg
    ) -> None:
        return self.server.recv_eventual_msg_without_reply(msg=msg)


def create_virtual_connection(node: AbstractNode):

    server = VirtualServerConnection(node=node)
    client = VirtualClientConnection(server=server)

    return client


class ObjectWithID(Serializable):
    """This object is the superclass for nearly all Syft objects. Subclassing
    from this object will cause an object to be initialized with a unique id
    using the process specified in the UID class.

    .. note::
        At the time of writing, the only class in Syft which doesn't have an ID
        of some kind is the Client class because it's job is to point to another
        object (which has an ID).

    .. note::
        Be aware of performance choices in this class because it is used so
        heavily across the entire codebase. Assume every method is going to
        be called thousands of times during the working day of an average
        data scientist using syft (and millions of times in the context of a
        machine learning job).

    """

    def __init__(self, id=None):
        """This initializer only exists to set the id attribute, which is the
        primary purpose of this class. It also sets the 'as_wrapper' flag
        for the 'Serializable' superclass.

        Args:
            id: an override which can be used to set an ID for this object

        """

        if id is None:
            id = randint(0, 1000)

        self._id = id

        # while this class is never used as a simple wrapper,
        # it's possible that sub-classes of this class will be.
        super().__init__()

    @property
    def id(self):
        """We reveal ObjectWithID.id as a property to discourage users and
        developers of Syft from modifying .id attributes after an object
        has been initialized.

        Returns:
            returns the unique id of the object
        """
        return self._id

    def __eq__(self, other) -> bool:
        """Checks to see if two ObjectWithIDs are actually the same object.

        This checks to see whether this ObjectWithIDs is equal to another by
        comparing whether they have the same .id objects. These objects
        come with their own __eq__ function which we assume to be correct.

        Args:
            other: this is the other ObjectWithIDs to be compared with

        Returns:
            True/False based on whether the objects are the same
        """

        try:
            return self.id == other.id
        except Exception:
            return False

    def __repr__(self) -> str:
        """
        Return a human-readable representation of the ObjectWithID with brackets
        so that it can be easily spotted when nested inside of the human-
        readable representations of other objects.

        Returns:
            a human-readable version of the ObjectWithID

        """

        no_dash = str(self.id.value).replace("-", "")
        return f"<{type(self).__name__}: {no_dash}>"

    def repr_short(self) -> str:
        """
        Return a SHORT human-readable version of the ID which
        makes it print nicer when embedded (often alongside other
        UID objects) within other object __repr__ methods.

        Returns:
            a SHORT human-readable version of SpecificLocation
        """

        return f"<{type(self).__name__}:{self.id.repr_short()}>"


class Route(ObjectWithID):
    def __init__(self, schema, stops=None):
        super().__init__()
        if stops is None:
            stops = list()
        self.schema = schema
        self.stops = stops

    @property
    def icon(self) -> str:
        return "🛣️ "

    @property
    def pprint(self) -> str:
        return f"{self.icon} ({self.class_name})"


class RouteSchema(ObjectWithID):
    """An object which contains the IDs of the origin node and
    set of the destination node. Multiple routes can subscribe
    to the same RouteSchema and routing, logic is thus split into
    two groups of functionality:

    1) Discovering new routes
    2) Comparing known routes to find the best one for a message
    """

    def __init__(self, destination):
        self.destination = destination


class SoloRoute(Route):
    def __init__(
        self,
        destination,
        connection,
    ) -> None:
        super().__init__(schema=RouteSchema(destination=destination))
        self.connection = connection

    def send_immediate_msg_without_reply(
        self, msg
    ) -> None:
        self.connection.send_immediate_msg_without_reply(msg=msg)

    def send_eventual_msg_without_reply(
        self, msg
    ) -> None:
        self.connection.send_eventual_msg_without_reply(msg=msg)

    def send_immediate_msg_with_reply(
        self, msg
    ):
        return self.connection.send_immediate_msg_with_reply(msg=msg)


class Node(AbstractNode):

    """
    Basic class for a syft node behavior, explicit purpose node will
    inherit this class (e.g., Device, Domain, Network, and VirtualMachine).

    Each node is identified by an id of type ID and a name of type string.
    """

    client_type = Client
    child_type_client_type = Client

    ChildT = "Node"
    child_type = ChildT

    signing_key = None
    verify_key = None

    def __init__(
        self,
        name=None,
        network=None,
        domain=None,
        device=None,
        vm=None,
        signing_key=None,
        verify_key=None,
        db_path=None,
    ):

        # The node has a name - it exists purely to help the
        # end user have some idea about what this node is in a human
        # readable form. It is not guaranteed to be unique (or to
        # really be anything for that matter).
        super().__init__(
            name=name, network=network, domain=domain, device=device, vm=vm
        )

        # We need to register all the services once a node is created
        # On the off chance someone forgot to do this (super unlikely)
        # this flag exists to check it.
        self.services_registered = False

        # In order to be able to write generic services (in .service)
        # which can work for all node types, sometimes we need to have
        # a reference to what node type this node is. This attribute
        # provides that ability.
        self.node_type = type(self).__name__
        self.immediate_msg_with_reply_router = {}

        # for messages which don't lead to a reply, this uses
        # the type of the message to look up the service
        # which addresses that message
        self.immediate_msg_without_reply_router = {}

        # for messages which don't need to be run right now
        # and will not generate a reply.
        self.eventual_msg_without_reply_router = {}

        # PERMISSION REGISTRY:
        self.root_verify_key = self.verify_key  # TODO: CHANGE
        self.guest_verify_key_registry = set()
        self.admin_verify_key_registry = set()
        self.cpl_ofcr_verify_key_registry = set()
        self.in_memory_client_registry = {}
        # TODO: remove hacky signaling_msgs when SyftMessages become Storable.
        self.signaling_msgs = {}

        # For logging the number of messages received
        self.message_counter = 0

    @property
    def icon(self):
        return "📍"

    def get_client(self, routes=None):
        if not routes:
            conn_client = create_virtual_connection(node=self)
            solo = SoloRoute(destination=self.target_id, connection=conn_client)
            # inject name
            setattr(solo, "name", f"Route ({self.name} <-> {self.name} Client)")
            routes = [solo]

        return self.client_type(  # type: ignore
            name=self.name,
            routes=routes,
            network=self.network,
            domain=self.domain,
            device=self.device,
            vm=self.vm,
            signing_key=None,  # DO NOT PASS IN A SIGNING KEY!!! The client generates one.
            verify_key=None,  # DO NOT PASS IN A VERIFY KEY!!! The client generates one.
        )

    def get_root_client(self, routes=None):
        client = self.get_client(routes=routes)
        return client

    @property
    def known_nodes(self):
        """This is a property which returns a list of all known node
        by returning the clients we used to interact with them from
        the object store."""
        return list(self.in_memory_client_registry.values())

    @property
    def known_child_nodes(self):
        if self.child_type_client_type is not None:
            return [
                client
                for client in self.in_memory_client_registry.values()
                if all(
                    [
                        self.network is None
                        or client.network is None
                        or self.network == client.network,
                        self.domain is None
                        or client.domain is None
                        or self.domain == client.domain,
                        self.device is None
                        or client.device is None
                        or self.device == client.device,
                        self.vm is None or client.vm is None or self.vm == client.vm,
                    ]
                )
            ]
        else:
            return []

    def recv_immediate_msg_with_reply(
        self, msg
    ):
        # exceptions can be easily triggered which break any WebRTC loops
        # so we need to catch them here and respond with a special exception
        # message reply
        try:
            # try to process message
            response = self.process_message(
                msg=msg, router=self.immediate_msg_with_reply_router
            )

        except Exception:
            pass

        res_msg = response.sign(signing_key=self.signing_key)  # type: ignore
        return res_msg

    def recv_immediate_msg_without_reply(
        self, msg
    ):
        self.process_message(msg=msg, router=self.immediate_msg_without_reply_router)

        return None

    def recv_eventual_msg_without_reply(
        self, msg
    ):
        self.process_message(msg=msg, router=self.eventual_msg_without_reply_router)

    # TODO: Add SignedEventualSyftMessageWithoutReply and others
    def process_message(
        self, msg, router: dict
    ):
        return None

    def ensure_services_have_been_registered_error_if_not(self):
        if not self.services_registered:
            raise (
                Exception(
                    "Please call _register_services on node. This seems to have"
                    "been skipped for some reason."
                )
            )

    def __repr__(self):
        no_dash = str(self.id).replace("-", "")
        return f"{self.node_type}: {self.name}: {no_dash}"


class Location(Serializable):
    """This represents the location of a node, including
    location-relevant metadata (such as how long it takes
    for us to communicate with this location, etc.)"""

    def __init__(self, name=None) -> None:
        if name is None:
            name = Serializable.random_name()
        self.name = name
        super().__init__()


class SpecificLocation(ObjectWithID, Location):
    """This represents the location of a single Node object
    represented by a single UID. It may not have any functionality
    beyond Location but there is logic, which interprets it differently."""

    def __init__(self, id=None, name=None):
        ObjectWithID.__init__(self, id=id)
        self.name = name if name is not None else self.name

    @property
    def icon(self) -> str:
        return "📌"

    @property
    def pprint(self) -> str:
        output = f"{self.icon} {self.name} ({self.class_name})@{self.id.emoji()}"
        return output


class VirtualMachineClient(Client):

    vm: SpecificLocation  # redefine the type of self.vm to not be optional
    import sympc as sympc_lib
    sympc = sympc_lib
    # import syft.lib.python
    class Asdf:
        def List(a):
            return [[0]] * 100
    python = Asdf

    def __init__(
        self,
        name,
        routes,
        vm,
        network=None,
        domain=None,
        device=None,
        signing_key=None,
        verify_key=None,
    ):
        super().__init__(
            name=name,
            routes=routes,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
        )

        self.post_init()

    @property
    def id(self):
        return self.vm.id

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.name}>"


class VirtualMachine(Node):
    client_type = VirtualMachineClient
    vm = None  # redefine the type of self.vm to not be optional
    signing_key = None
    verify_key = None
    child_type_client_type = None
    # import sympc as sympc_library
    sympc = None

    def __init__(
        self,
        *,  # Trasterisk
        name=None,
        network=None,
        domain=None,
        device=None,
        vm=SpecificLocation(),
        signing_key=None,
        verify_key=None,
    ):
        super().__init__(
            name=name,
            network=network,
            domain=domain,
            device=device,
            vm=vm,
            signing_key=signing_key,
            verify_key=verify_key,
        )

        # specific location with name
        self.vm = SpecificLocation(name=self.name)

    @property
    def id(self):
        return self.vm.id

    def message_is_for_me(self, msg):
        # this needs to be defensive by checking vm_id NOT vm.id or it breaks
        try:
            return msg.address.vm_id == self.id
        except Exception:
            return False
