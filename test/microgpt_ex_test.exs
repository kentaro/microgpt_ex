defmodule MicrogptExTest do
  use ExUnit.Case
  doctest MicrogptEx

  test "greets the world" do
    assert MicrogptEx.hello() == :world
  end
end
